import cv2
import numpy as np

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.5


def alignImages(im1, im2):

    # Detect ORB features and  descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width,channel= im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg




e1 = cv2.getTickCount()

if __name__ == '__main__':

  # Read images
  im = cv2.imread("after.png", cv2.IMREAD_COLOR)
  imReference = cv2.imread("before.png", cv2.IMREAD_COLOR)

  #removing background
  lower_red = np.array([140, 60, 150])
  upper_red = np.array([360, 360, 360])
  lower_blue = np.array([98, 0, 205])
  upper_blue = np.array([122, 59, 360])

  hsv_before = cv2.cvtColor(imReference, cv2.COLOR_BGR2HSV)
  hsv_after = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
  # maskes
  mask_red1 = cv2.inRange(hsv_before, lower_red, upper_red)
  mask_red2 = cv2.inRange(hsv_after, lower_red, upper_red)
  mask_blue1 = cv2.inRange(hsv_before, lower_blue, upper_blue)
  mask_blue2 = cv2.inRange(hsv_after, lower_blue, upper_blue)
  whole_before = cv2.bitwise_or(mask_red1, mask_blue1)
  whole_after = cv2.bitwise_or(mask_red2, mask_blue2)
  #two images after removing background
  before = cv2.bitwise_and(imReference, imReference, mask=whole_before)
  after = cv2.bitwise_or(im, im, mask=whole_after)

 #align taken img
  beforeAligned = alignImages(before, after)
  hsv = cv2.cvtColor(beforeAligned, cv2.COLOR_BGR2HSV)
  red1Aligned = cv2.inRange(hsv, lower_red, upper_red)
  blue1Aligned = cv2.inRange(hsv,lower_blue, upper_blue)
  whole1Aligned = cv2.bitwise_or(red1Aligned,blue1Aligned)


 #extracct parts of change
  growth_death = cv2.absdiff(whole_after, whole1Aligned)
  growth_recovered = cv2.absdiff(mask_red2, red1Aligned)
  whole_white = cv2.bitwise_or(mask_blue2, blue1Aligned)

 # to get the new part
  both_img = cv2.bitwise_or(whole_after, whole1Aligned)
  newPart = cv2.absdiff(both_img, whole1Aligned)
  newPart = cv2.medianBlur(newPart,15)

# Dead part
  midStep3 = cv2.absdiff(both_img, newPart)
  deadPart_mid = cv2.absdiff(both_img, whole_after)
  deadPart = cv2.medianBlur(deadPart_mid,15)
  #recovered
  recovered_bleatched = cv2.absdiff(growth_death,growth_recovered)

  recovered = cv2.bitwise_and(recovered_bleatched,mask_red2)
  recovered = cv2.medianBlur(recovered,15)

  #bleatched
  bleatched_mid = cv2.absdiff(whole_white, blue1Aligned)
  bleatched=cv2.medianBlur(bleatched_mid,15)


  #countour
  contoursRec, _ = cv2.findContours(recovered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  contoursNew, _ = cv2.findContours(newPart, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  contoursBlet, _ = cv2.findContours(bleatched, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  contoursDead, _ = cv2.findContours(deadPart, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

  #rectangle grawing od recovered

  for cnt in contoursRec:
     area = cv2.contourArea(cnt)
     if area > 300:
         x, y, w, h = cv2.boundingRect(cnt)
         # box margin
         x = x-5
         y = y-10
         w = w+12
         h = h+13
         cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

  for cnt in contoursBlet:
     area = cv2.contourArea(cnt)
     if area > 300:
         x, y, w, h = cv2.boundingRect(cnt)
         x = x-5
         y = y-10
         w = w+12
         h = h+13
         cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)


  for cnt in contoursDead:
     area = cv2.contourArea(cnt)
     if area > 300:
         x, y, w, h = cv2.boundingRect(cnt)
         x = x-5
         y = y-10
         w = w+12
         h = h+13
         cv2.rectangle(im, (x, y), (x + w, y + h), (89, 255, 255), 2)


  for cnt in contoursNew:
     area = cv2.contourArea(cnt)
     if area > 300:
         x, y, w, h = cv2.boundingRect(cnt)
         x = x-5
         y = y-10
         w = w+12
         h = h+13
         cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

  cv2.imshow('im',im)
  e2 = cv2.getTickCount()
  time = (e2-e1)/cv2.getTickFrequency()
  print(time)

  cv2.waitKey(0)
  cv2.destroyAllWindows()