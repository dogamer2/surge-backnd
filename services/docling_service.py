class DoclingService:
    def __init__(self):
        self.converter = None

    def _get_converter(self):
        if self.converter is not None:
            return self.converter

        try:
            from docling.document_converter import (
                DocumentConverter,
                PdfFormatOption,
                PowerpointFormatOption,
                WordFormatOption,
            )
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
        except ImportError as exc:
            raise RuntimeError(
                "Docling is not installed. Enable it to parse PDF/DOCX/PPTX documents."
            ) from exc

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False

        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PPTX, InputFormat.PDF, InputFormat.DOCX],
            format_options={
                InputFormat.DOCX: WordFormatOption(
                    pipeline_options=pipeline_options,
                ),
                InputFormat.PPTX: PowerpointFormatOption(
                    pipeline_options=pipeline_options,
                ),
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            },
        )
        return self.converter

    def parse_to_markdown(self, file_path: str) -> str:
        converter = self._get_converter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown()
