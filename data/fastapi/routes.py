from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def home() -> dict[str, str]:
    return {"message": "hello"}


@router.get("/items/{item_id}")
def get_item(item_id: int) -> dict[str, int]:
    return {"item_id": item_id}
