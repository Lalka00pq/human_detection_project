# project
from src.schemas.service_output import GetCheckModelLoaded
# 3rdparty
from fastapi import APIRouter, Request

router = APIRouter(tags=["Model Loaded Check"])


@router.get(
    "/check_model_loaded",
    summary="Проверяет загружена ли модель",
    description="Проверяет загружена ли модель",
    response_model=GetCheckModelLoaded,
    response_description="Датакласс с информацией о загруженности модели",
)
def check_model_loaded(request: Request) -> GetCheckModelLoaded:
    """Проверяет загружена ли модель

    Args:
        request (Request): Объект запроса

    Returns:
        GetCheckModelLoaded: Датакласс с информацией о загруженности модели
    """
    if request.app.state.model is None:
        return GetCheckModelLoaded(
            Model_loaded=False,
        )
    return GetCheckModelLoaded(
        Model_loaded=True,
    )
