from django.urls import path
from django.http import JsonResponse
import numpy as np
import cv2
import base64 
import json
from PIL import Image
from io import BytesIO
from django.views.decorators.csrf import csrf_exempt 
def rectCounter(coutours):
    rectCon = []
    for i in coutours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    return rectCon
def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]  # [0,0]
    myPointsNew[3] = myPoints[np.argmax(add)]  # [w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # [w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)]  # [h,0]
    return myPointsNew
def splitBoxes(img, rows=10, cols=4):
    row_height = img.shape[0] // rows
    col_width = img.shape[1] // cols
    boxes = []
    for r in range(rows):
        for c in range(cols):
            box = img[r * row_height:(r + 1) * row_height, c * col_width:(c + 1) * col_width]
            boxes.append(box)
    return boxes
def checking(image, answer):
    path = image
    width = 700
    height = 700
    questions = 20
    choices = 4
    img1 = cv2.imread(path)
    img = img1.copy()
    img = cv2.resize(img, (width, height))
    imgContours = img.copy()
    imgFinal = cv2.resize(img, (700, 700)) 
    imgBiggestContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 5, 50)
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
    rectCon = rectCounter(contours)
    grading = []
    counter0 = getCornerPoints(rectCon[0])
    counter1 = getCornerPoints(rectCon[1])
    counters = (counter0, counter1)
    if counter0[0][0][0] > counter1[0][0][0]:
        counters = (counter1, counter0)
    n = 0
    for counter in counters:
        questions1 = questions // 2
        # print(counter[0][0],)
        if counter.size != 0:
            cv2.drawContours(imgBiggestContours, counter, -1, (0, 255, 0), 30)
            counter = reorder(counter)
            pts1 = np.float32(counter)
            pts2 = np.float32([[0, 0], [250, 0], [0, 600], [250, 600]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (250, 600))
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 150, 255, cv2.THRESH_BINARY_INV)[1]
            boxes = splitBoxes(imgThresh)
            myPixelVal = np.zeros((questions1, choices))
            countC = 0
            countR = 0
            for image in boxes:
                totalPixel = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixel
                countC += 1
                if countC == choices:
                    countR += 1
                    countC = 0
            myIndex = []
            for x in range(0, questions1):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                if 1300 > np.amax(arr):
                    myIndex.append(None)
                else:
                    myIndex.append(myIndexVal[0][0])
            for i in range(0, questions1):
                if myIndex[i] == answer[i + 10 * n]:
                    grading.append(1)
                elif myIndex[i] is None:
                    grading.append(None)
                else:
                    grading.append(0)          
                    
                         
            imgDrawing = np.zeros_like(imgWarpColored)
            for i in range(questions1):
                if myIndex[i] is not None:
                    x = (myIndex[i] * imgDrawing.shape[1]) // choices
                    y = (i * imgDrawing.shape[0]) // questions1
                    w = imgDrawing.shape[1] // choices
                    h = imgDrawing.shape[0] // questions1
                    color = (0, 181, 26) if myIndex[i] == answer[i + 10 * n] else (0, 0, 255)
                    cv2.rectangle(imgDrawing, (x, y), (x + w, y + h), color, 3)
            
            matrix_2 = cv2.getPerspectiveTransform(pts2, pts1)
            img_warp = cv2.warpPerspective(imgDrawing, matrix_2, (700, 700))
            imgFinal = cv2.addWeighted(imgFinal, 1, img_warp, 1, 0)
            # cv2.imwrite('aaa.png',imgFinal)
            n +=1
    _, buffer = cv2.imencode('.png', imgFinal)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    # print(grading)
    return grading, img_base64

# answers = [0, 2, 1, 3, 2, 2, 0, 3, 1, 2, 2, 1, 1, 3, 0, 1, 2, 2, 3, 1] 
@csrf_exempt
def app(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')

            if image_data:
                img_code = Image.open(BytesIO(base64.b64decode(image_data)))
                img_code.save('core/test.png', "PNG")
                img = 'core/test.png'
                grading = checking(img)  # Assuming 'checking' is a function you have defined elsewhere
                count_true = grading.count(1)
                count_false = grading.count(0)
                count_none = grading.count(None)
                score = count_true / len(grading) * 100 if len(grading) > 0 else 0
                return JsonResponse({'Correct': count_true, 'F': count_false, 'N': count_none, 'D': score})
            else:
                return JsonResponse({'error': 'No image provided'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    

@csrf_exempt
def createApi(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            answers = data.get('answers') 
            image_data = data.get('image')

            if image_data:
                # Decode the image data
                format, imgstr = image_data.split(';base64,')
                img_code = Image.open(BytesIO(base64.b64decode(imgstr)))

                # Save the image
                img_code.save('core/test.png', "PNG")
                # Process the image
                img = 'core/test.png'
                grading, encode = checking(img, answers)  
                count_true = grading.count(1)
                count_false = grading.count(0)
                count_none = grading.count(None)
                score = count_true / len(grading) * 100 if len(grading) > 0 else 0

                return JsonResponse({'Correct': count_true, 'F': count_false, 'N': count_none, 'D': score, 'encode' : encode})

            else:
                return JsonResponse({'error': 'No image  provided'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
@csrf_exempt
def gett(request):
    if request == 'GET':
        return JsonResponse({'error': 'Invalid request method'})
    else:
        return JsonResponse({'ER': 'Invalid request method'})
urlpatterns = [
    path('get', gett),
    path('', createApi), 
    path('api/', app)
]