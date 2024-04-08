
var track_conf = {
    "comment": "// 多目标跟踪的配置参数",
    "version": "1.0",
    "main_category": 1,
    "sub_category": 1,
    "max_track_num": 100,
    "max_track_length": 50,
    "min_confidence": 0.1
}

var infer_conf = {
    "comment":
    {
        "device": [-1, 0, 1, 2, 3],
        "backend": ["ONNX"],
        "resize_type": ["Stretch", "LetterBox", "Fill"],
        "interp_mode": ["Linear", "Nearest", "Cubic"],
        "channel_format": ["RGB", "BGR"],
        "dtype": ["float32", "uint8"],
        "input_format": ["NCHW", "NHWC"],
        "preprocess": ["ImageType", "VideoType", "FeatureType"],
        "postprocess": ["YoloGrid", "...comsumer-defined..."]
    },
    "model":
    {
        "version": "1.0",
        "model_name": "yolov7_tiny_grid",
        "encrypt": false,
        "max_batch_size": 16,
        "device": 1,
        "model_file": "../models/yolov7/yolov7-grid.onnx",
        "weight_file": "../models/yolov7/yolov7-grid.onnx",
        "backend": "TRT",
        "dynamic_batch": false,
        "enable_int8": false,
        "enable_fp16": true
    },
    "preprocess":
    {
        "ImageType":
            [
                {
                    "output_name": "images",
                    "param":
                    {
                        "resize_type": "Stretch",
                        "output_diims": [1, 3, 640, 640],
                        "output_format": "NCHW",
                        "output_dtype": "float32",

                        "interp_mode": "Linear",
                        "letterbox_color": [114, 114, 114],
                        "channel_format": "RGB",

                        "normalize": true,
                        "mean": [0, 0, 0],
                        "std": [255.0, 255.0, 255.0]
                    }
                }
            ],
        "VideoType":
            [],
        "FeatureType":
            []
    },

    "infer":
    {
        "input_names": ["images"],
        "input_dims": [[1, 3, 640, 640]],
        "output_names": ["output"]
    },

    "postprocess":
    {
        "YoloGrid":
        {
            "input_names": ["output"],
            "output_data_type": "DetectResult",
            "main_category": [1, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                10, 3, 3, 7, 4, 4, 4, 4, 4, 4,
                4, 4, 4, 4, 9, 9, 9, 9, 9, 5,
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                7, 7, 7, 7, 7, 7, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
                7, 7, 7, 8, 8, 8, 8, 8, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            "sub_category": [1, 6, 1, 5, 8, 3, 10, 2, 9, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "num_classes": 80,
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4

        }

    }

}


var nodeCounter = 1;

function addNodeToDrawFlow(NodeType, pos_x, pos_y) {
    if (editor.editor_mode === 'fixed') {
        return false;
    }
    pos_x = pos_x * (editor.precanvas.clientWidth / (editor.precanvas.clientWidth * editor.zoom)) - (editor.precanvas.getBoundingClientRect().x * (editor.precanvas.clientWidth / (editor.precanvas.clientWidth * editor.zoom)));
    pos_y = pos_y * (editor.precanvas.clientHeight / (editor.precanvas.clientHeight * editor.zoom)) - (editor.precanvas.getBoundingClientRect().y * (editor.precanvas.clientHeight / (editor.precanvas.clientHeight * editor.zoom)));
    var nodeName = NodeType + nodeCounter;

    var html = '';
    var data = {};
    var output = [];
    var intNum = 0;
    var multiple = false;
    switch (NodeType) {
        case 'ImageSrc':
            html = `<div class="title-box"><i class="fas fa-image"></i> ${nodeName}</div>`;
            data = {};
            output = ["Frame"];
            intNum = 0;
            multiple = false;
            break;
        case 'VideoSrc':
            html = ` <div class="title-box"><i class="fas fa-video"></i>  ${nodeName}</div> `;
            data = {};
            output = ["Frame"];
            intNum = 0;
            multiple = false;
            break;
        case 'Infer':
            html = `<div class="title-box"><i class="fas fa-brain"></i>  ${nodeName}</div> `;
            data = infer_conf;
            output = ["DetectResult"];
            intNum = 1;
            multiple = true;
            break;
        case 'Track':
            html = `<div class="title-box"><i class="fas fa-route"></i> ${nodeName}</div>`;
            data = track_conf;
            output = ["DetectResult"];
            intNum = 1;
            multiple = false;
            break;
        case 'Base':
            html = `<div class="title-box"><i class="fas fa-lightbulb"></i> ${nodeName}</div> `;
            data = {}
            output = ["PipelineResult"];
            intNum = 1;
            multiple = false;
            break;
        default:
    }

    editor.addNode(nodeName, intNum, output.length, pos_x, pos_y, NodeType, data, html, output, multiple);
    nodeCounter++;
}