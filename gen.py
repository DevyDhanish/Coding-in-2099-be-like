import cv2
  
vid = cv2.VideoCapture(0)
dataset_dir = "dataset/"

i = 0

while(True):
    ret, frame = vid.read()
    
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('taking photos', grayscale)
    cv2.imwrite(f"{dataset_dir}{i}.jpg", grayscale)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i += 1

vid.release()
cv2.destroyAllWindows()