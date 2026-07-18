class PCCError(Exception):
    """Base class for expected prototype failures."""


class InvalidArgument(PCCError):
    pass


class InvalidPack(PCCError):
    pass


class PackMismatch(PCCError):
    pass


class CapacityExceeded(PCCError):
    pass


class FrameError(PCCError):
    pass


class AuthenticationError(PCCError):
    pass


class TruncatedTranscript(PCCError):
    pass
