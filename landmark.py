from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    
    if not exif_data:
        return None
    
    # Extract the EXIF tags and GPS data
    exif_dict = {}
    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        exif_dict[tag_name] = value

    # Get GPS data
    if 'GPSInfo' in exif_dict:
        gps_info = exif_dict['GPSInfo']
        gps_data = {}
        for t in gps_info:
            sub_tag = GPSTAGS.get(t, t)
            gps_data[sub_tag] = gps_info[t]
        return gps_data
    else:
        return None

def convert_to_decimal(degrees, minutes, seconds, direction):
    # Extract the numerator and denominator from the IFDRational objects
    degrees_decimal = float(degrees.numerator) / float(degrees.denominator)  # numerator / denominator
    minutes_decimal = float(minutes.numerator) / float(minutes.denominator) / 60  # numerator / denominator / 60
    seconds_decimal = float(seconds.numerator) / float(seconds.denominator) / 3600  # numerator / denominator / 3600
    
    # Final decimal value
    decimal = degrees_decimal + minutes_decimal + seconds_decimal
    
    if direction in ['S', 'W']:  # Negative for South or West
        decimal = -decimal
    
    return decimal

def get_coordinates(gps_info):
    if gps_info:
        latitude = gps_info.get('GPSLatitude')
        latitude_ref = gps_info.get('GPSLatitudeRef')
        longitude = gps_info.get('GPSLongitude')
        longitude_ref = gps_info.get('GPSLongitudeRef')

        if latitude and longitude and latitude_ref and longitude_ref:
            # Convert the GPS coordinates to decimal degrees
            lat = convert_to_decimal(latitude[0], latitude[1], latitude[2], latitude_ref)
            lon = convert_to_decimal(longitude[0], longitude[1], longitude[2], longitude_ref)

            return lat, lon
    return None

# Example usage
image_path = 'all_data/image.jpeg'  # Update with your image path
gps_info = get_exif_data(image_path)
coordinates = get_coordinates(gps_info)

if coordinates:
    print(f"Latitude: {coordinates[0]}, Longitude: {coordinates[1]}")
else:
    print("No GPS data found in EXIF.")
