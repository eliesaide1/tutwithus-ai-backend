"""Build the signed booking_contract emitted on confirmation."""

from typing import Any

from utils.apexchat.schemas.models import BookingState


def build_contract(bs: BookingState) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "description": bs.description or f"{bs.level_name} {bs.subject_name} session",
        "tutorId": bs.tutor_id,
        "subjectId": bs.subject_id,
        "levelId": bs.level_id,
        "curriculum": bs.curriculum_code,
        "date": bs.date,
        "timeFrom": bs.time_from,
        "timeTo": bs.time_to,
        # "materialUploaded": bs.material_uploaded,
    }
    if bs.invitees:
        payload["invitees"] = [
            {"email": inv.email, "firstName": inv.firstName, "lastName": inv.lastName}
            for inv in bs.invitees
        ]
    return {
        "action": "book_session",
        "ready_to_book": True,
        "booking_payload": payload,
    }
