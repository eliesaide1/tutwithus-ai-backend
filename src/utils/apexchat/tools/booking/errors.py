"""Typed errors for the booking flow."""


class BookingError(Exception):
    """Base for all recoverable booking-flow errors. Carries a user-safe message."""

    def __init__(self, user_message: str, *, code: str = "booking_error", detail: dict | None = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.code = code
        self.detail = detail or {}


class InvalidTransitionError(BookingError):
    """Raised when code attempts a state-machine transition that isn't allowed."""

    def __init__(self, from_step, to_step):
        super().__init__(
            user_message=(
                f"I can't move to a step of the booking process that doesn't follow from the current one."
                "let's continue from where we left off."
            ),
            code="invalid_transition",
            detail={"from": from_step.value, "to": to_step.value},
        )


class StepDataError(BookingError):
    """Raised when the data needed to present a step (levels, slots, ...) is unavailable."""


class BusinessRuleViolation(BookingError):
    """Raised when a business rule blocks progression (e.g. slot too soon)."""


class SlotConflict(BookingError):
    """Raised when the chosen slot is no longer available at confirmation time."""

    def __init__(self, message: str = "That slot was just taken — please pick another."):
        super().__init__(user_message=message, code="slot_conflict")
