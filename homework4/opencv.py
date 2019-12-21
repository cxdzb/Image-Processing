import cv2  # 导入opencv库


# 参数：（要寻找的目标，原图片）
def templatex(img_target, img_root):
    # 模板匹配方法
    # toolx=cv2.TM_SQDIFF_NORMED
    toolx = cv2.TM_CCORR_NORMED
    # toolx=cv2.TM_CCOEFF_NORMED
    h, w = img_target.shape[:2]  # 获取目标图像的高和宽
    # 操作匹配
    result = cv2.matchTemplate(img_root, img_target, toolx)
    # 得到区域
    min_x, max_x, min_y, max_y = cv2.minMaxLoc(result)
    # 获取起始点坐标
    if toolx == cv2.TM_SQDIFF_NORMED:
        tl = min_y
    else:
        tl = max_y
    # 获取结束点坐标，其中tl[0]表示起始点x轴值，tl[1]表示y
    br = (tl[0] + w, tl[1] + h)
    # 创建一个矩形框，参数（要写到的图片，起始点坐标，结束点坐标，颜色值，厚度）
    cv2.rectangle(img_root, tl, br, (0, 0, 255), 5)
    # 显示图片
    cv2.imshow("img_rootxx", img_root)


# 读取一张图片，地址不能带中文
imgviewx = cv2.imread("./resource/image1.png")
# 创建一个窗口，中文显示会出乱码
cv2.namedWindow("标题", cv2.WINDOW_NORMAL)
# 获取原图片截图
areax = imgviewx[110:529, 778:1200]
cv2.imshow("jjjttt", areax)
templatex(areax, imgviewx)
# 显示图片，参数：（窗口标识字符串，imread读入的图像）
cv2.imshow("标题", imgviewx)
# 窗口等待任意键盘按键输入,0为一直等待,其他数字为毫秒数
cv2.waitKey(0)
# 销毁窗口，退出程序
cv2.destroyAllWindows()
