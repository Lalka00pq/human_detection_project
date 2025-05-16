# project
from src.schemas.service_output import GetCheckModelLoaded
# 3rdparty
from fastapi import APIRouter, Request

router = APIRouter(tags=["Model Loaded Check"])


@router.get(
    "/check_model_loaded",
    summary="Проверяет загружена ли модель",
    description="Проверяет загружена ли модель",
    response_description="Датакласс с информацией о загруженности модели",
)
async def check_model_loaded(request: Request):
    """Проверяет загружена ли модель

    Args:
        request (Request): Объект запроса

    Returns:
        GetCheckModelLoaded: Датакласс с информацией о загруженности модели
    """
    if request.app.state.model is None:
        return False
    return True
