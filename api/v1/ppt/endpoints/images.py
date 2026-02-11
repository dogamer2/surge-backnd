from typing import List
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from models.image_prompt import ImagePrompt
from models.sql.image_asset import ImageAsset
from services.credit_service import consume_credits, get_request_user_id
from services.database import get_async_session
from services.image_generation_service import ImageGenerationService
from utils.asset_directory_utils import get_images_directory
import os
import uuid
from utils.file_utils import get_file_name_with_random_uuid

IMAGES_ROUTER = APIRouter(prefix="/images", tags=["Images"])


def to_public_image_path(path_value: str) -> str:
    if not path_value:
        return path_value
    if path_value.startswith("http") or path_value.startswith("/static/") or path_value.startswith("/app_data/"):
        return path_value

    normalized = path_value.replace("\\", "/")
    filename = os.path.basename(normalized)
    return f"/app_data/images/{filename}"


def to_disk_image_path(path_value: str) -> str:
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    if path_value.startswith("/app_data/images/"):
        filename = os.path.basename(path_value)
        return os.path.join(get_images_directory(), filename)
    return os.path.join(get_images_directory(), os.path.basename(path_value))


@IMAGES_ROUTER.get("/generate")
async def generate_image(
    request: Request,
    prompt: str,
    sql_session: AsyncSession = Depends(get_async_session),
):
    user_id = get_request_user_id(request)
    await consume_credits(sql_session, user_id, "image")
    images_directory = get_images_directory()
    image_prompt = ImagePrompt(prompt=prompt)
    image_generation_service = ImageGenerationService(images_directory)

    image = await image_generation_service.generate_image(image_prompt)
    if not isinstance(image, ImageAsset):
        return image

    sql_session.add(image)
    await sql_session.commit()

    return to_public_image_path(image.path)


@IMAGES_ROUTER.get("/generated", response_model=List[ImageAsset])
async def get_generated_images(sql_session: AsyncSession = Depends(get_async_session)):
    try:
        images = await sql_session.scalars(
            select(ImageAsset)
            .where(ImageAsset.is_uploaded == False)
            .order_by(ImageAsset.created_at.desc())
        )
        image_list = list(images)
        for image in image_list:
            image.path = to_public_image_path(image.path)
        return image_list
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve generated images: {str(e)}"
        )


@IMAGES_ROUTER.post("/upload")
async def upload_image(
    file: UploadFile = File(...), sql_session: AsyncSession = Depends(get_async_session)
):
    try:
        new_filename = get_file_name_with_random_uuid(file)
        image_path = os.path.join(
            get_images_directory(), os.path.basename(new_filename)
        )

        with open(image_path, "wb") as f:
            f.write(await file.read())

        image_asset = ImageAsset(path=image_path, is_uploaded=True)

        sql_session.add(image_asset)
        await sql_session.commit()

        image_asset.path = to_public_image_path(image_asset.path)
        return image_asset
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")


@IMAGES_ROUTER.get("/uploaded", response_model=List[ImageAsset])
async def get_uploaded_images(sql_session: AsyncSession = Depends(get_async_session)):
    try:
        images = await sql_session.scalars(
            select(ImageAsset)
            .where(ImageAsset.is_uploaded == True)
            .order_by(ImageAsset.created_at.desc())
        )
        image_list = list(images)
        for image in image_list:
            image.path = to_public_image_path(image.path)
        return image_list
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve uploaded images: {str(e)}"
        )


@IMAGES_ROUTER.delete("/{id}", status_code=204)
async def delete_uploaded_image_by_id(
    id: uuid.UUID, sql_session: AsyncSession = Depends(get_async_session)
):
    try:
        # Fetch the asset to get its actual file path
        image = await sql_session.get(ImageAsset, id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        disk_path = to_disk_image_path(image.path)
        if os.path.exists(disk_path):
            os.remove(disk_path)

        await sql_session.delete(image)
        await sql_session.commit()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {str(e)}")
