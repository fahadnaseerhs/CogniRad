#!/usr/bin/env python3
"""
Generate QR code for CogniRad deployment URL.
Usage: python generate_qr.py https://your-app-url.com
"""

import sys
import qrcode
from pathlib import Path


def generate_qr_code(url: str, output_file: str = "cognirad_qr.png"):
    """Generate a QR code for the given URL."""
    
    # Create QR code instance
    qr = qrcode.QRCode(
        version=1,  # Controls size (1 is smallest, auto-adjusts)
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
        box_size=10,  # Size of each box in pixels
        border=4,  # Border size in boxes
    )
    
    # Add data
    qr.add_data(url)
    qr.make(fit=True)
    
    # Create image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save
    img.save(output_file)
    print(f"✓ QR code generated: {output_file}")
    print(f"✓ URL: {url}")
    print(f"\nStudents can scan this QR code to access CogniRad!")
    
    return output_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_qr.py <URL>")
        print("\nExample:")
        print("  python generate_qr.py https://cognirad.onrender.com")
        print("\nOr install qrcode first:")
        print("  pip install qrcode[pil]")
        sys.exit(1)
    
    url = sys.argv[1]
    
    # Validate URL
    if not url.startswith(("http://", "https://")):
        print(f"Warning: URL should start with http:// or https://")
        print(f"Adding https:// prefix...")
        url = f"https://{url}"
    
    try:
        generate_qr_code(url)
    except ImportError:
        print("\n❌ Error: qrcode library not installed")
        print("\nInstall it with:")
        print("  pip install qrcode[pil]")
        sys.exit(1)


if __name__ == "__main__":
    main()
