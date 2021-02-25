from darknet import parse_args, DETECT



args = parse_args()

image_detector = DETECT(args.filepath, b"cfg/yolov3.cfg", b"yolov3.weights", b"cfg/coco.data")

objects = ["car", "truck", "person"]
image = "data/cars.jpg"

image_detector.parse_image(image, objects, args.debug)