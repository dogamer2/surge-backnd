import os
import asyncio
import base64
import re
from datetime import datetime
from typing import Optional, List, Dict
from uuid import UUID
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends
from pydantic import BaseModel
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold, HarmBlockThreshold
from google.genai.errors import APIError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func
from utils.asset_directory_utils import get_images_directory
from services.database import get_async_session
from models.sql.presentation_layout_code import PresentationLayoutCodeModel
from .prompts import (
    GENERATE_HTML_SYSTEM_PROMPT,
    HTML_TO_REACT_SYSTEM_PROMPT,
    HTML_EDIT_SYSTEM_PROMPT,
)
from models.sql.template import TemplateModel
from utils.llm_provider import is_google_selected, get_llm_provider
from utils.get_env import get_google_api_key_env


# Create separate routers for each functionality
SLIDE_TO_HTML_ROUTER = APIRouter(prefix="/slide-to-html", tags=["slide-to-html"])
HTML_TO_REACT_ROUTER = APIRouter(prefix="/html-to-react", tags=["html-to-react"])
HTML_EDIT_ROUTER = APIRouter(prefix="/html-edit", tags=["html-edit"])
LAYOUT_MANAGEMENT_ROUTER = APIRouter(
    prefix="/template-management", tags=["template-management"]
)


# Request/Response models for slide-to-html endpoint
class SlideToHtmlRequest(BaseModel):
    image: str  # Partial path to image file (e.g., "/app_data/images/uuid/slide_1.png")
    xml: str  # OXML content as text
    fonts: Optional[List[str]] = None  # Optional normalized root fonts for this slide


class SlideToHtmlResponse(BaseModel):
    success: bool
    html: str


# Request/Response models for html-edit endpoint
class HtmlEditResponse(BaseModel):
    success: bool
    edited_html: str
    message: Optional[str] = None


# Request/Response models for html-to-react endpoint
class HtmlToReactRequest(BaseModel):
    html: str  # HTML content to convert to React component
    image: Optional[str] = None  # Optional image path to provide visual context


class HtmlToReactResponse(BaseModel):
    success: bool
    react_component: str
    message: Optional[str] = None


# Request/Response models for layout management endpoints
class LayoutData(BaseModel):
    presentation: UUID  # UUID of the presentation
    layout_id: str  # Unique identifier for the layout
    layout_name: str  # Display name of the layout
    layout_code: str  # TSX/React component code for the layout
    fonts: Optional[List[str]] = None  # Optional list of font links


class SaveLayoutsRequest(BaseModel):
    layouts: list[LayoutData]


class SaveLayoutsResponse(BaseModel):
    success: bool
    saved_count: int
    message: Optional[str] = None


class GetLayoutsResponse(BaseModel):
    success: bool
    layouts: list[LayoutData]
    message: Optional[str] = None
    template: Optional[dict] = None
    fonts: Optional[List[str]] = None


class PresentationSummary(BaseModel):
    presentation_id: UUID
    layout_count: int
    last_updated_at: Optional[datetime] = None
    template: Optional[dict] = None


class GetPresentationSummaryResponse(BaseModel):
    success: bool
    presentations: List[PresentationSummary]
    total_presentations: int
    total_layouts: int
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    detail: str
    error_code: Optional[str] = None


class TemplateCreateRequest(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None


class TemplateCreateResponse(BaseModel):
    success: bool
    template: dict
    message: Optional[str] = None


class TemplateInfo(BaseModel):
    id: UUID
    name: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None


def _is_retryable_google_error(error_message: str) -> bool:
    msg = (error_message or "").lower()
    return any(
        token in msg
        for token in [
            "502",
            "503",
            "504",
            "bad gateway",
            "service unavailable",
            "deadline exceeded",
            "timeout",
            "temporarily unavailable",
            "429",
            "rate limit",
        ]
    )


async def _generate_content_with_retry(client: genai.Client, model: str, contents, operation: str):
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config={"response_mime_type": "text/plain"},
            )
        except Exception as e:
            last_error = e
            error_msg = str(e)
            if not _is_retryable_google_error(error_msg) or attempt == 2:
                raise
            delay_seconds = 1 * (2 ** attempt)
            print(
                f"{operation} transient Google GenAI error on attempt {attempt + 1}/3: {error_msg}. Retrying in {delay_seconds}s..."
            )
            await asyncio.sleep(delay_seconds)
    if last_error:
        raise last_error


async def generate_html_from_slide(
    base64_image: str,
    media_type: str,
    xml_content: str,
    api_key: str,
    fonts: Optional[List[str]] = None,
) -> str:
    """
    Generate HTML content from slide image and XML using Google GenAI (Gemini) API.
    """

    print("Generating HTML from slide image and XML using Google GenAI API...")

    try:
        if not is_google_selected():
            raise HTTPException(
                status_code=500,
                detail="Google provider not selected",
            )

        # ✅ Correct new SDK usage
        from google import genai

        client = genai.Client(api_key=api_key)

        fonts_text = (
            f"\nFONTS (Normalized root families used in this slide, use where required): {', '.join(fonts)}"
            if fonts
            else ""
        )

        response = await _generate_content_with_retry(
            client=client,
            model="gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": GENERATE_HTML_SYSTEM_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": media_type,
                                "data": base64_image,
                            }
                        },
                        {"text": f"OXML:\n{xml_content}{fonts_text}"},
                    ],
                }
            ],
            operation="HTML generation",
        )

        html_content = getattr(response, "text", None) or ""

        print(f"Received HTML content length: {len(html_content)}")

        if not html_content:
            raise HTTPException(
                status_code=500,
                detail="No HTML content generated by Google GenAI",
            )

        return html_content

    except Exception as e:
        error_msg = str(e)
        print(f"Google GenAI error: {error_msg}")

        if "invalid_argument" in error_msg.lower() or "invalid argument" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Google GenAI invalid request during HTML generation: {error_msg}",
            )
        elif "timeout" in error_msg.lower():
            raise HTTPException(
                status_code=408,
                detail=f"Google GenAI timeout during HTML generation: {error_msg}",
            )
        elif "connection" in error_msg.lower() or _is_retryable_google_error(error_msg):
            raise HTTPException(
                status_code=502,
                detail=f"Google GenAI connection error during HTML generation: {error_msg}",
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Google GenAI API error during HTML generation: {error_msg}",
            )



        data_url = f"data:{media_type};base64,{base64_image}"
        fonts_text = (
            f"\nFONTS (Normalized root families used in this slide, use where it is required): {', '.join(fonts)}"
            if fonts
            else ""
        )
        user_text = f"OXML: \n\n{fonts_text}"
        
        # Prepare content parts for Google GenAI
        content_parts = [
            {"text": GENERATE_HTML_SYSTEM_PROMPT},
            {
                "inline_data": {
                    "mime_type": media_type,
                    "data": base64_image
                }
            },
            {"text": user_text}
        ]

        print("Making Google GenAI request for HTML generation...")
        response = client.generate_content(content_parts)

        # Extract the response text
        html_content = ""
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and len(candidate.content) > 0:
                html_content = candidate.content[0].text

        print(f"Received HTML content length: {len(html_content)}")

        if not html_content:
            raise HTTPException(
                status_code=500, detail="No HTML content generated by Google GenAI"
            )

        return html_content

    except APIError as e:
        print(f"Google GenAI API Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Google GenAI API error during HTML generation: {str(e)}"
        )
    except Exception as e:
        # Handle various API errors
        error_msg = str(e)
        print(f"Exception occurred: {error_msg}")
        print(f"Exception type: {type(e)}")
        if "timeout" in error_msg.lower():
            raise HTTPException(
                status_code=408,
                detail=f"Google GenAI API timeout during HTML generation: {error_msg}",
            )
        elif "connection" in error_msg.lower():
            raise HTTPException(
                status_code=502,
                detail=f"Google GenAI API connection error during HTML generation: {error_msg}",
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Google GenAI API error during HTML generation: {error_msg}",
            )
    except Exception as e:
        # Handle various API errors
        error_msg = str(e)
        print(f"Exception occurred: {error_msg}")
        print(f"Exception type: {type(e)}")
        if "timeout" in error_msg.lower():
            raise HTTPException(
                status_code=408,
                detail=f"OpenAI API timeout during HTML generation: {error_msg}",
            )
        elif "connection" in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail=f"OpenAI API connection error during HTML generation: {error_msg}",
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API error during HTML generation: {error_msg}",
            )


async def generate_react_component_from_html(
    html_content: str,
    api_key: str,
    image_base64: Optional[str] = None,
    media_type: Optional[str] = None,
) -> str:
    """
    Convert HTML content to TSX React component using Google GenAI (Gemini) API.

    Args:
        html_content: Generated HTML content
        api_key: Google API key
        image_base64: Optional base64 image for context
        media_type: MIME type of the image

    Returns:
        Generated TSX React component as string

    Raises:
        HTTPException: If API call fails or no content is generated
    """
    def sanitize_jsx(react_code: str) -> str:
        # Replace class -> className
        react_code = re.sub(r'\bclass=', 'className=', react_code)
        # Normalize malformed self-closing tags generated by LLMs (e.g. "<img ... / />")
        react_code = re.sub(r'/\s*/\s*>', '/>', react_code)
        react_code = re.sub(r'\s+/>\s*>', '/>', react_code)
        # Self-close img/br/hr/input only when not already self-closed
        react_code = re.sub(r'<(img|br|hr|input)([^>]*?)(?<!/)>', r'<\1\2 />', react_code)
        # Remove any leading "json" lines
        react_code = re.sub(r'^\s*json\s*', '', react_code, flags=re.MULTILINE)
        # Remove markdown fences
        react_code = react_code.replace("```tsx", "").replace("```", "")
        return react_code

    try:
        client = genai.Client(api_key=api_key)

        # Build a single user turn; Gemini does not accept "system" role here.
        content_parts = [{"text": f"{HTML_TO_REACT_SYSTEM_PROMPT}\n\nHTML INPUT:\n{html_content}"}]
        if image_base64 and media_type:
            content_parts.insert(0, {"inline_data": {"mime_type": media_type, "data": image_base64}})

        response = await _generate_content_with_retry(
            client=client,
            model="gemini-2.5-flash-lite",
            contents=[
                {
                    "role": "user",
                    "parts": content_parts,
                }
            ],
            operation="React generation",
        )

        react_content = getattr(response, "text", None) or ""
        if not react_content:
            raise HTTPException(
                status_code=500,
                detail="No React component generated by Google GenAI",
            )

        # Sanitize JSX
        react_content = sanitize_jsx(react_content)

        # Remove import/export lines
        filtered_lines = [
            line for line in react_content.split("\n")
            if not (line.strip().startswith("import ") or line.strip().startswith("export "))
        ]
        return "\n".join(filtered_lines)

    except APIError as e:
        print(f"Google GenAI API Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Google GenAI API error during React generation: {str(e)}",
        )
    except Exception as e:
        error_msg = str(e)
        print(f"Exception occurred: {error_msg}")
        if "invalid_argument" in error_msg.lower() or "invalid argument" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Google GenAI invalid request during React generation: {error_msg}",
            )
        elif "timeout" in error_msg.lower():
            raise HTTPException(status_code=408, detail=f"Google GenAI timeout: {error_msg}")
        elif "connection" in error_msg.lower() or _is_retryable_google_error(error_msg):
            raise HTTPException(status_code=502, detail=f"Google GenAI connection error: {error_msg}")
        else:
            raise HTTPException(status_code=500, detail=f"Google GenAI error: {error_msg}")


async def edit_html_with_images(
    current_ui_base64: str,
    sketch_base64: Optional[str],
    media_type: str,
    html_content: str,
    prompt: str,
    api_key: str,
) -> str:
    """
    Edit HTML content based on one or two images and a text prompt using OpenAI GPT-5 Responses API.

    Args:
        current_ui_base64: Base64 encoded current UI image data
        sketch_base64: Base64 encoded sketch/indication image data (optional)
        media_type: MIME type of the images (e.g., 'image/png')
        html_content: Current HTML content to edit
        prompt: Text prompt describing the changes
        api_key: OpenAI API key

    Returns:
        Edited HTML content as string

    Raises:
        HTTPException: If API call fails or no content is generated
    """
    try:
        client = genai.Client(api_key=api_key)

        print("Making Responses API request for HTML editing...")

        current_data_url = f"data:{media_type};base64,{current_ui_base64}"
        sketch_data_url = (
            f"data:{media_type};base64,{sketch_base64}" if sketch_base64 else None
        )

        content_parts = [
            {"type": "input_image", "image_url": current_data_url},
            {
                "type": "input_text",
                "text": f"CURRENT HTML TO EDIT:\n{html_content}\n\nTEXT PROMPT FOR CHANGES:\n{prompt}",
            },
        ]
        if sketch_data_url:
            # Insert sketch image after current UI image for context
            content_parts.insert(
                1, {"type": "input_image", "image_url": sketch_data_url}
            )

        input_payload = [
            {"role": "system", "content": HTML_EDIT_SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]

        response = client.responses.create(
            model="gpt-5",
            input=input_payload,
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
        )

        edited_html = (
            getattr(response, "output_text", None)
            or getattr(response, "text", None)
            or ""
        )

        print(f"Received edited HTML content length: {len(edited_html)}")

        if not edited_html:
            raise HTTPException(
                status_code=500,
                detail="No edited HTML content generated by OpenAI GPT-5",
            )

        return edited_html

    except APIError as e:
        print(f"OpenAI API Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"OpenAI API error during HTML editing: {str(e)}"
        )
    except Exception as e:
        # Handle various API errors
        error_msg = str(e)
        print(f"Exception occurred: {error_msg}")
        print(f"Exception type: {type(e)}")
        if "timeout" in error_msg.lower():
            raise HTTPException(
                status_code=408,
                detail=f"OpenAI API timeout during HTML editing: {error_msg}",
            )
        elif "connection" in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail=f"OpenAI API connection error during HTML editing: {error_msg}",
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API error during HTML editing: {error_msg}",
            )


# ENDPOINT 1: Slide to HTML conversion
@SLIDE_TO_HTML_ROUTER.post("/", response_model=SlideToHtmlResponse)
async def convert_slide_to_html(request: SlideToHtmlRequest):
    """
    Convert a slide image and its OXML data to HTML using Anthropic Claude API.

    Args:
        request: JSON request containing image path and XML content

    Returns:
        SlideToHtmlResponse with generated HTML
    """
    try:
        # Get API key based on selected provider
        if is_google_selected():
            api_key = get_google_api_key_env()
            if not api_key:
                raise HTTPException(
                    status_code=500, detail="GOOGLE_API_KEY environment variable not set"
                )
        else:
            # Fallback to OpenAI if Google not selected (keeping backward compatibility)
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise HTTPException(
                    status_code=500, detail="OPENAI_API_KEY environment variable not set"
                )

        # Resolve image path to actual file system path
        image_path = request.image

        # Handle different path formats
        if image_path.startswith("/app_data/images/"):
            # Remove the /app_data/images/ prefix and join with actual images directory
            relative_path = image_path[len("/app_data/images/") :]
            actual_image_path = os.path.join(get_images_directory(), relative_path)
        elif image_path.startswith("/static/"):
            # Handle static files
            relative_path = image_path[len("/static/") :]
            actual_image_path = os.path.join("static", relative_path)
        else:
            # Assume it's already a full path or relative to images directory
            if os.path.isabs(image_path):
                actual_image_path = image_path
            else:
                actual_image_path = os.path.join(get_images_directory(), image_path)

        # Check if image file exists
        if not os.path.exists(actual_image_path):
            raise HTTPException(
                status_code=404, detail=f"Image file not found: {image_path}"
            )

        # Read and encode image to base64
        with open(actual_image_path, "rb") as image_file:
            image_content = image_file.read()
        base64_image = base64.b64encode(image_content).decode("utf-8")

        # Determine media type from file extension
        file_extension = os.path.splitext(actual_image_path)[1].lower()
        media_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(file_extension, "image/png")

        # Generate HTML using the extracted function
        html_content = await generate_html_from_slide(
            base64_image=base64_image,
            media_type=media_type,
            xml_content=request.xml,
            api_key=api_key,
            fonts=request.fonts,
        )

        html_content = html_content.replace("```html", "").replace("```", "")

        return SlideToHtmlResponse(success=True, html=html_content)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full error for debugging
        print(f"Unexpected error during slide to HTML processing: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing slide to HTML: {str(e)}"
        )


# ENDPOINT 2: HTML to React component conversion
@HTML_TO_REACT_ROUTER.post("/", response_model=HtmlToReactResponse)
async def convert_html_to_react(request: HtmlToReactRequest):
    """
    Convert HTML content to TSX React component using Anthropic Claude API.

    Args:
        request: JSON request containing HTML content

    Returns:
        HtmlToReactResponse with generated React component
    """
    try:
        # Get Google API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500, detail="GOOGLE_API_KEY environment variable not set"
            )

        # Validate HTML content
        if not request.html or not request.html.strip():
            raise HTTPException(status_code=400, detail="HTML content cannot be empty")

        # Optionally resolve image and encode to base64
        image_b64 = None
        media_type = None
        if request.image:
            image_path = request.image
            if image_path.startswith("/app_data/images/"):
                relative_path = image_path[len("/app_data/images/") :]
                actual_image_path = os.path.join(get_images_directory(), relative_path)
            elif image_path.startswith("/static/"):
                relative_path = image_path[len("/static/") :]
                actual_image_path = os.path.join("static", relative_path)
            else:
                actual_image_path = (
                    image_path
                    if os.path.isabs(image_path)
                    else os.path.join(get_images_directory(), image_path)
                )
            if os.path.exists(actual_image_path):
                with open(actual_image_path, "rb") as f:
                    image_b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(actual_image_path)[1].lower()
                media_type = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }.get(ext, "image/png")

        # Convert HTML to React component
        react_component = await generate_react_component_from_html(
            html_content=request.html,
            api_key=api_key,
            image_base64=image_b64,
            media_type=media_type,
        )

        react_component = react_component.replace("```tsx", "").replace("```", "")

        return HtmlToReactResponse(
            success=True,
            react_component=react_component,
            message="React component generated successfully",
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full error for debugging
        print(f"Unexpected error during HTML to React processing: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing HTML to React: {str(e)}"
        )


# ENDPOINT 3: HTML editing with images
@HTML_EDIT_ROUTER.post("/", response_model=HtmlEditResponse)
async def edit_html_with_images_endpoint(
    current_ui_image: UploadFile = File(..., description="Current UI image file"),
    sketch_image: Optional[UploadFile] = File(
        None, description="Sketch/indication image file (optional)"
    ),
    html: str = Form(..., description="Current HTML content to edit"),
    prompt: str = Form(..., description="Text prompt describing the changes"),
):
    """
    Edit HTML content based on one or two uploaded images and a text prompt using Anthropic Claude API.

    Args:
        current_ui_image: Uploaded current UI image file
        sketch_image: Uploaded sketch/indication image file (optional)
        html: Current HTML content to edit (form data)
        prompt: Text prompt describing the changes (form data)

    Returns:
        HtmlEditResponse with edited HTML
    """
    try:
        # Get OpenAI API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500, detail="GOOGLE_API_KEY environment variable not set"
            )

        # Validate inputs
        if not html or not html.strip():
            raise HTTPException(status_code=400, detail="HTML content cannot be empty")

        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="Text prompt cannot be empty")

        # Validate current UI image file
        if (
            not current_ui_image.content_type
            or not current_ui_image.content_type.startswith("image/")
        ):
            raise HTTPException(
                status_code=400, detail="Current UI file must be an image"
            )

        # Validate sketch image file only if provided
        if sketch_image and (
            not sketch_image.content_type
            or not sketch_image.content_type.startswith("image/")
        ):
            raise HTTPException(status_code=400, detail="Sketch file must be an image")

        # Read and encode current UI image to base64
        current_ui_content = await current_ui_image.read()
        current_ui_base64 = base64.b64encode(current_ui_content).decode("utf-8")

        # Read and encode sketch image to base64 only if provided
        sketch_base64 = None
        if sketch_image:
            sketch_content = await sketch_image.read()
            sketch_base64 = base64.b64encode(sketch_content).decode("utf-8")

        # Use the content type from the uploaded files
        media_type = current_ui_image.content_type

        # Edit HTML using the function
        edited_html = await edit_html_with_images(
            current_ui_base64=current_ui_base64,
            sketch_base64=sketch_base64,
            media_type=media_type,
            html_content=html,
            prompt=prompt,
            api_key=api_key,
        )

        edited_html = edited_html.replace("```html", "").replace("```", "")

        return HtmlEditResponse(
            success=True, edited_html=edited_html, message="HTML edited successfully"
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full error for debugging
        print(f"Unexpected error during HTML editing: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing HTML editing: {str(e)}"
        )


# ENDPOINT 4: Save layouts for a presentation
@LAYOUT_MANAGEMENT_ROUTER.post(
    "/save-templates",
    response_model=SaveLayoutsResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def save_layouts(
    request: SaveLayoutsRequest, session: AsyncSession = Depends(get_async_session)
):
    """
    Save multiple layouts for presentations.

    Args:
        request: JSON request containing array of layout data
        session: Database session

    Returns:
        SaveLayoutsResponse with success status and count of saved layouts

    Raises:
        HTTPException: 400 for validation errors, 500 for server errors
    """
    try:
        # Validate request data
        if not request.layouts:
            raise HTTPException(status_code=400, detail="Layouts array cannot be empty")

        if len(request.layouts) > 50:  # Reasonable limit
            raise HTTPException(
                status_code=400, detail="Cannot save more than 50 layouts at once"
            )

        saved_count = 0

        for i, layout_data in enumerate(request.layouts):
            # Validate individual layout data
            if (
                not layout_data.presentation
                or not str(layout_data.presentation).strip()
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"Layout {i+1}: presentation_id cannot be empty",
                )

            if not layout_data.layout_id or not layout_data.layout_id.strip():
                raise HTTPException(
                    status_code=400, detail=f"Layout {i+1}: layout_id cannot be empty"
                )

            if not layout_data.layout_name or not layout_data.layout_name.strip():
                raise HTTPException(
                    status_code=400, detail=f"Layout {i+1}: layout_name cannot be empty"
                )

            if not layout_data.layout_code or not layout_data.layout_code.strip():
                raise HTTPException(
                    status_code=400, detail=f"Layout {i+1}: layout_code cannot be empty"
                )

            # Check if layout already exists for this presentation and layout_id
            stmt = select(PresentationLayoutCodeModel).where(
                PresentationLayoutCodeModel.presentation == layout_data.presentation,
                PresentationLayoutCodeModel.layout_id == layout_data.layout_id,
            )
            result = await session.execute(stmt)
            existing_layout = result.scalar_one_or_none()

            if existing_layout:
                # Update existing layout
                existing_layout.layout_name = layout_data.layout_name
                existing_layout.layout_code = layout_data.layout_code
                existing_layout.fonts = layout_data.fonts
                existing_layout.updated_at = datetime.now()
            else:
                # Create new layout
                new_layout = PresentationLayoutCodeModel(
                    presentation=layout_data.presentation,
                    layout_id=layout_data.layout_id,
                    layout_name=layout_data.layout_name,
                    layout_code=layout_data.layout_code,
                    fonts=layout_data.fonts,
                )
                session.add(new_layout)

            saved_count += 1

        await session.commit()

        return SaveLayoutsResponse(
            success=True,
            saved_count=saved_count,
            message=f"Successfully saved {saved_count} layout(s)",
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        print(f"Unexpected error saving layouts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while saving layouts: {str(e)}",
        )


# ENDPOINT 5: Get layouts for a presentation
@LAYOUT_MANAGEMENT_ROUTER.get(
    "/get-templates/{presentation}",
    response_model=GetLayoutsResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid presentation ID"},
        404: {
            "model": ErrorResponse,
            "description": "No layouts found for presentation",
        },
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_layouts(
    presentation: UUID, session: AsyncSession = Depends(get_async_session)
):
    """
    Retrieve all layouts for a specific presentation.

    Args:
        presentation: UUID of the presentation
        session: Database session

    Returns:
        GetLayoutsResponse with layouts data

    Raises:
        HTTPException: 404 if no layouts found, 400 for invalid UUID, 500 for server errors
    """
    try:
        # Validate presentation_id format (basic UUID check)
        if not presentation or len(str(presentation).strip()) == 0:
            raise HTTPException(
                status_code=400, detail="Presentation ID cannot be empty"
            )

        # Query layouts for the given presentation_id
        stmt = select(PresentationLayoutCodeModel).where(
            PresentationLayoutCodeModel.presentation == presentation
        )
        result = await session.execute(stmt)
        layouts_db = result.scalars().all()

        # Check if any layouts were found
        if not layouts_db:
            raise HTTPException(
                status_code=404,
                detail=f"No layouts found for presentation ID: {presentation}",
            )

        # Convert to response format
        layouts = [
            LayoutData(
                presentation=layout.presentation,
                layout_id=layout.layout_id,
                layout_name=layout.layout_name,
                layout_code=layout.layout_code,
                fonts=layout.fonts,
            )
            for layout in layouts_db
        ]

        # Aggregate unique fonts across all layouts
        aggregated_fonts: set[str] = set()
        for layout in layouts_db:
            if layout.fonts:
                aggregated_fonts.update([f for f in layout.fonts if isinstance(f, str)])
        fonts_list = sorted(list(aggregated_fonts)) if aggregated_fonts else None

        # Fetch template meta
        template_meta = await session.get(TemplateModel, presentation)
        template = None
        if template_meta:
            template = {
                "id": template_meta.id,
                "name": template_meta.name,
                "description": template_meta.description,
                "created_at": template_meta.created_at,
            }

        return GetLayoutsResponse(
            success=True,
            layouts=layouts,
            message=f"Retrieved {len(layouts)} layout(s) for presentation {presentation}",
            template=template,
            fonts=fonts_list,
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Error retrieving layouts for presentation {presentation}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while retrieving layouts: {str(e)}",
        )


# ENDPOINT: Get all presentations with layout counts
@LAYOUT_MANAGEMENT_ROUTER.get(
    "/summary",
    response_model=GetPresentationSummaryResponse,
    summary="Get all presentations with layout counts",
    description="Retrieve a summary of all presentations and the number of layouts in each",
    responses={
        200: {
            "model": GetPresentationSummaryResponse,
            "description": "Presentations summary retrieved successfully",
        },
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_presentations_summary(
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get summary of all presentations with their layout counts.
    """
    try:
        # Query to get presentation_id, count of layouts, and MAX(updated_at)
        stmt = select(
            PresentationLayoutCodeModel.presentation,
            func.count(PresentationLayoutCodeModel.id).label("layout_count"),
            func.max(PresentationLayoutCodeModel.updated_at).label("last_updated_at"),
        ).group_by(PresentationLayoutCodeModel.presentation)

        result = await session.execute(stmt)
        presentation_data = result.all()

        # Convert to response format with template info if available
        presentations = []
        for row in presentation_data:
            try:
                template_meta = await session.get(TemplateModel, row.presentation)
                template = None
                if template_meta:
                    template = {
                        "id": str(template_meta.id),
                        "name": template_meta.name,
                        "description": template_meta.description,
                        "created_at": template_meta.created_at.isoformat() if template_meta.created_at else None,
                    }
                
                # Ensure last_updated_at is properly formatted
                last_updated = row.last_updated_at
                if last_updated and hasattr(last_updated, 'isoformat'):
                    last_updated = last_updated.isoformat()
                
                presentations.append(
                    PresentationSummary(
                        presentation_id=row.presentation,
                        layout_count=row.layout_count,
                        last_updated_at=last_updated,
                        template=template,
                    )
                )
            except Exception as row_error:
                print(f"Warning: Error processing presentation row {row.presentation}: {str(row_error)}")
                # Still add the row but without template info
                presentations.append(
                    PresentationSummary(
                        presentation_id=row.presentation,
                        layout_count=row.layout_count,
                        last_updated_at=row.last_updated_at,
                        template=None,
                    )
                )

        # Calculate totals
        total_presentations = len(presentations)
        total_layouts = sum(p.layout_count for p in presentations)

        return GetPresentationSummaryResponse(
            success=True,
            presentations=presentations,
            total_presentations=total_presentations,
            total_layouts=total_layouts,
            message=f"Retrieved {total_presentations} presentation(s) with {total_layouts} total layout(s)",
        )

    except Exception as e:
        print(f"Error retrieving presentations summary: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while retrieving presentations summary: {str(e)}",
        )


@LAYOUT_MANAGEMENT_ROUTER.post(
    "/templates",
    response_model=TemplateCreateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def create_template(
    request: TemplateCreateRequest,
    session: AsyncSession = Depends(get_async_session),
):
    try:
        if not request.id or not request.name:
            raise HTTPException(status_code=400, detail="id and name are required")

        # Upsert template by id
        existing = await session.get(TemplateModel, request.id)
        if existing:
            existing.name = request.name
            existing.description = request.description
        else:
            session.add(
                TemplateModel(
                    id=request.id, name=request.name, description=request.description
                )
            )
        await session.commit()

        # Read back
        template = await session.get(TemplateModel, request.id)
        return TemplateCreateResponse(
            success=True,
            template={
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "created_at": template.created_at,
            },
            message="Template saved",
        )
    except HTTPException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to save template: {str(e)}"
        )


@LAYOUT_MANAGEMENT_ROUTER.delete("/delete-templates/{template_id}", status_code=204)
async def delete_template(
    template_id: UUID,
    session: AsyncSession = Depends(get_async_session),
):
    try:
        await session.execute(
            delete(TemplateModel).where(TemplateModel.id == template_id)
        )
        await session.execute(
            delete(PresentationLayoutCodeModel).where(
                PresentationLayoutCodeModel.presentation == template_id,
            )
        )
        await session.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete template")
