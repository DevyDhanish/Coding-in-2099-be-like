import cv2
import os

TAKEPHOTO : bool = False

def getPhotoConfirmation():
    take_photo = input("Take photo (y/n) --> ")

    if take_photo == "y" or take_photo == "Y":
        return True
    
    elif take_photo == "n" or take_photo == "N":
        return False
    
    else:
        print("Invalid input")
        return getPhotoConfirmation()


vid = cv2.VideoCapture(0)
dataset_dir = "dataset/"

# get general info about data
data_name = input("Enter data name ( this will be used a the folder name aswell ) --> ")
max_dataset = int(input("Max Photo to take for this data --> "))
print("Smile for the photo!")

# create the folder for the photo
os.mkdir(os.path.join(dataset_dir, data_name))

TAKEPHOTO = getPhotoConfirmation()

counter = 0
while(counter <= max_dataset):
    ret, img = vid.read()
    
    cv2.imshow('taking photos', img)
    
    if TAKEPHOTO:
        cv2.imwrite(f"{os.path.join(dataset_dir, data_name, str(counter))}.jpg", img)
        print(f"Took photo!!, saved as {os.path.join(dataset_dir, data_name, str(counter))}.jpg")

    cv2.waitKey(1)
    
    counter += 1

vid.release()
cv2.destroyAllWindows()

print("Done !!")