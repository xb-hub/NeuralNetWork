import cv2
import mahotas
import numpy as np

class ImageProcess:
    def __init__(self):
        self.imagePath = "../image/number_1.jpg"
        self.dx = 15
        self.dy = 15

    # 定义一个缩放函数
    def resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        # 高度模式
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        # 宽度模式
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    # 进行旋转变换
    def deskew(self, image, width):
        (h, w) = image.shape[:2]
        moments = cv2.moments(image)
        skew = moments["mu11"] / moments["mu02"]
        M = np.float32([
            [1, skew, -0.5 * w * skew],
            [0, 1, 0]])
        image = cv2.warpAffine(image, M, (w, h),
                               flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        image = self.resize(image, width=width)
        return image

    # 把数字缩放到图片中心
    def center_extent(self, image, size):
        (eW, eH) = size

        # 如果宽度>高度
        if image.shape[1] > image.shape[0]:
            image = self.resize(image, width=eW)
        else:
            image = self.resize(image, height=eH)

        extent = np.zeros((eH, eW), dtype="uint8")
        offsetX = (eW - image.shape[1]) // 2
        offsetY = (eH - image.shape[0]) // 2
        extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image

        # 计算图片的质量中心
        (cY, cX) = np.round(mahotas.center_of_mass(extent)).astype("int32")
        (dX, dY) = ((size[0] // 2) - cX, (size[1] // 2) - cY)
        M = np.float32([[1, 0, dX], [0, 1, dY]])
        # 把质量中心移动到图片的中心
        extent = cv2.warpAffine(extent, M, size)

        return extent

    def process(self, network):
        # 加载被分类的图片
        image_process = ImageProcess()
        image = cv2.imread(self.imagePath)
        size = image.shape
        image = cv2.resize(image, (int(size[1] / 5), int(size[0] / 5)))
        # 图片预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        # 根据轮廓对数字进行切分
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key=lambda x: x[1])
        index = 0
        for (c, _) in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            # 对于一定大小的数字才进行识别
            if w >= 1 and h >= 5:
                # 提取ROI区域
                roi = gray[y - self.dy:y + h + self.dy, x - self.dx:x + w + self.dx]
                thresh = roi.copy()
                # 智能识别阈值
                T = mahotas.thresholding.otsu(roi)
                thresh[thresh > T] = 255
                # 过滤掉颜色更亮的背景
                thresh = cv2.bitwise_not(thresh)
                thresh = image_process.deskew(thresh, network.config.image_width)
                thresh = image_process.center_extent(thresh, (network.config.image_width, network.config.image_height))
                # 测试预处理效果
                thresh = thresh.reshape(28 * 28)

                # 根据模型来预测输出
                digit = network.query(thresh)
                print("I think that number is: {}".format(digit))

                # 把识别出的数字用绿色框显示出来
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # 在识别出来的框左上角标注数字
                cv2.putText(image, str(digit), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("image", image)
        cv2.imwrite("../image/detector_result.jpg", image)
        cv2.waitKey(0)

