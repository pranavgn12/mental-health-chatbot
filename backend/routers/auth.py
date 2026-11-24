from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from database import get_db
from models import User, Profile
from schemas.profile import RegisterRequest
from auth_utils import create_access_token

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@router.post("/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(
        (User.email == request.email) | (User.username == request.username)
    ).first()
    if existing_user:
        if existing_user.email == request.email:
            raise HTTPException(status_code=400, detail="Email already registered")
        raise HTTPException(status_code=400, detail="Username already taken")

    user = User(
        username=request.username.strip(),
        email=request.email.strip().lower(),
        password_hash=pwd_context.hash(request.password),
    )
    try:
        db.add(user)
        db.flush()
        if request.profile is not None:
            db.add(Profile(user_id=user.id, **request.profile.model_dump()))
        db.commit()
        db.refresh(user)
    except Exception:
        db.rollback()
        raise

    return {"message": "User registered successfully", "user_id": user.id}


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not pwd_context.verify(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"user_id": user.id, "email": user.email})
    return {"access_token": token, "token_type": "bearer"}
