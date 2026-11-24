from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from auth_utils import get_current_user
from database import get_db
from models import Profile
from schemas.profile import ProfilePayload

router = APIRouter()


def _user_id(current_user):
    user_id = current_user.get("user_id") if isinstance(current_user, dict) else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user session")
    return int(user_id)


def _serialize(profile):
    if profile is None:
        return {
            "name": None, "date_of_birth": None, "gender": None,
            "occupation_status": None, "degree": None, "branch": None,
            "study_year": None, "job_title": None, "city": None,
            "state": None, "country": None,
        }
    return {
        "name": profile.name,
        "date_of_birth": profile.date_of_birth.isoformat() if profile.date_of_birth else None,
        "gender": profile.gender,
        "occupation_status": profile.occupation_status,
        "degree": profile.degree,
        "branch": profile.branch,
        "study_year": profile.study_year,
        "job_title": profile.job_title,
        "city": profile.city,
        "state": profile.state,
        "country": profile.country,
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }


@router.get("/me")
def get_my_profile(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    profile = db.query(Profile).filter(Profile.user_id == _user_id(current_user)).first()
    return {"profile": _serialize(profile)}


@router.put("/me")
def update_my_profile(
    payload: ProfilePayload,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id = _user_id(current_user)
    profile = db.query(Profile).filter(Profile.user_id == user_id).first()
    values = payload.model_dump()
    if profile is None:
        profile = Profile(user_id=user_id, **values)
        db.add(profile)
    else:
        for field, value in values.items():
            setattr(profile, field, value)
    db.commit()
    db.refresh(profile)
    return {"message": "Profile updated successfully", "profile": _serialize(profile)}
