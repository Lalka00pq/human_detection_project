# 3rdparty
from fastapi import APIRouter, Request

router = APIRouter(tags=["Model Loaded Check"])


@router.get(
    "/check_model_loaded",
    summary="Проверяет загружена ли модель",
    description="Проверяет загружена ли модель",
)
def check_model_loaded(request: Request):
    """Проверяет загружена ли модель

    Args:
        request (Request): Объект запроса

    Returns:
        _type_: _description_
    """
    if request.app.state.model is None:
        return {
            "Model_loaded": False,
        }
    return {
        "Model_loaded": True,
    }
