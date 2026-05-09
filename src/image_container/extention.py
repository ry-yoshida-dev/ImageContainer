from enum import Enum

class Extention(Enum):
    """
    Extention of the image.
    """
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    TIFF = "tiff"
    TIF = "tif"
    BMP = "bmp"
    HEIC = "heic"
    GIF = "gif"
    WEBP = "webp"
    AVIF = "avif"

    @property
    def is_lossless(self) -> bool:
        """
        Return whether the extension is treated as lossless by default.

        Returns
        -------
        bool
            True when the extension is commonly used with lossless encoding.
        """
        match self:
            case Extention.PNG | Extention.TIFF | Extention.TIF | Extention.BMP | Extention.GIF:
                return True
            case Extention.JPG | Extention.JPEG | Extention.HEIC | Extention.WEBP | Extention.AVIF:
                return False

    @property
    def is_transparency_supported(self) -> bool:
        """
        Return whether the extension commonly supports transparency.

        Returns
        -------
        bool
            True when alpha/transparency is commonly supported.
        """
        match self:
            case Extention.PNG | Extention.GIF | Extention.HEIC | Extention.WEBP | Extention.AVIF:
                return True
            case Extention.JPG | Extention.JPEG | Extention.TIFF | Extention.TIF | Extention.BMP:
                return False
