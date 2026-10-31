from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/register")
def register(
    request: RegisterRequest
):
    return {
        "message": "Register endpoint"
    }


@router.post("/login")
def login(
    request: LoginRequest
):
    return {
        "message": "Login endpoint"
    }