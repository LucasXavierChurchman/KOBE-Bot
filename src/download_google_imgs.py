from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

# the 'jump shot' keyword has a space to avoid downloading the fortnite skin images
# the directory is later changed to 'jumpshot' manually
arguments = {'keywords':'dunk, jump shot',
            'limit': 510,
            'print_urls':True,
            'output_directory': '../data/google_imgs/',
            'format': 'jpg',
            'type': 'photo'}   #creating list of arguments

paths = response.download(arguments)   #passing the arguments to the function
print(paths)