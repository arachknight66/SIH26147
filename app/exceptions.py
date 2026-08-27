class SignalInputError(Exception):
    """Base error with an actionable, user-facing message."""


class UnsupportedFileFormatError(SignalInputError): pass
class UnsupportedIQDatatypeError(SignalInputError): pass
class InvalidSampleCountError(SignalInputError): pass
class InvalidEndianError(SignalInputError): pass
class InvalidWavHeaderError(SignalInputError): pass
class CorruptSigMFMetadataError(SignalInputError): pass
class AmbiguousIQInterpretationError(SignalInputError): pass
