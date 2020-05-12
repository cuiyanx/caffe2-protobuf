class Caffe2ModelUtils {
  constructor(predictModel, initModel, isQuantized=false) {
    this._predict = predictModel;
    this._init = initModel;
    this._quantized = isQuantized;
    this._predictDataFormat = false;
    this._initDataFormat = false;

    this._checkDataFormat();
    this._initMap = this._initModelHandler();
    this._predictMap = this._predictModelHandler();
  }

  getCaffe2Model () {
    return this._predictMap;
  }

  getCaffe2InitModel () {
    return this._initMap;
  }

  _initModelHandler () {
    let initTensorMap = [];

    for (let op of this._init.op) {
      initTensorMap[op.output] = [];

      for (let arg of op.arg) {
        initTensorMap[op.output][arg.name] = [];
        let data = this._checkArgData(arg);
        initTensorMap[op.output][arg.name]["type"] = data.type;
        initTensorMap[op.output][arg.name]["value"] = data.value;
      }

      // uint8 => int8
      if (this._quantized &&
          typeof initTensorMap[op.output]["values"] != "undefined" &&
          initTensorMap[op.output]["values"]["type"] == "uint8" &&
          typeof initTensorMap[op.output]["Y_zero_point"] != "undefined" &&
          initTensorMap[op.output]["Y_zero_point"]["value"] == "128") {
        initTensorMap[op.output]["values"]["type"] = "int8";
        let tmpArray = [];
        for ( let val of Object.values(initTensorMap[op.output]["values"]["value"])) {
          tmpArray.push(val - 128);
        }
        initTensorMap[op.output]["values"]["value"] = tmpArray;
        initTensorMap[op.output]["Y_zero_point"]["value"] = "0";
      }

      // NCHW => NHWC
      if (this._initDataFormat &&
          typeof initTensorMap[op.output]["shape"] != "undefined" &&
          typeof initTensorMap[op.output]["values"] != "undefined") {
        if (initTensorMap[op.output]["shape"]["value"].length == 4) {
          initTensorMap[op.output] = this._NCHWtoNHWCforTensor(initTensorMap[op.output], true);
        } else {
          initTensorMap[op.output] = this._NCHWtoNHWCforTensor(initTensorMap[op.output], false);
        }
      }
    }

    return initTensorMap;
  }

  _predictModelHandler() {
    let predictTensorMap = [];

    for (let opIdx in this._predict.op) {
      predictTensorMap[opIdx] = [];
      let op = this._predict.op[opIdx];

      // name, type, engine
      predictTensorMap[opIdx]["name"] = op.name;
      predictTensorMap[opIdx]["operator"] = op.type;
      predictTensorMap[opIdx]["engine"] = op.engine;

      // input
      predictTensorMap[opIdx]["input"] = [];
      for (let inputIdx in op.input) {
        predictTensorMap[opIdx]["input"][inputIdx] = [];
        let input = op.input[inputIdx];
        predictTensorMap[opIdx]["input"][inputIdx]["name"] = input;

        for (let key in this._initMap[input]) {
          predictTensorMap[opIdx]["input"][inputIdx][key] = this._initMap[input][key];
        }
      }

      // output
      predictTensorMap[opIdx]["output"] = [];
      for (let outputIdx in op.output) {
        predictTensorMap[opIdx]["output"][outputIdx] = [];
        let output = op.output[outputIdx];
        predictTensorMap[opIdx]["output"][outputIdx]["name"] = output;
      }

      // arg
      predictTensorMap[opIdx]["arg"] = [];
      for (let argIdx in op.arg) {
        let arg = op.arg[argIdx];

        predictTensorMap[opIdx]["arg"][arg.name] = [];
        let data = this._checkArgData(arg);
        predictTensorMap[opIdx]["arg"][arg.name]["type"] = data.type;
        predictTensorMap[opIdx]["arg"][arg.name]["value"] = data.value;

        if (arg.name == "order") {
          predictTensorMap[opIdx]["arg"][arg.name]["type"] = "str";
          let orderTmp = [];
          for (let val of predictTensorMap[opIdx]["arg"][arg.name]["value"]) {
            orderTmp.push(String.fromCharCode(val));
          }
          predictTensorMap[opIdx]["arg"][arg.name]["value"] = orderTmp.join("");
        }

        // uint8 => int8
        if (this._quantized && arg.name != "order" &&
            typeof predictTensorMap[opIdx]["arg"][arg.name]["type"] != "undefined" &&
            predictTensorMap[opIdx]["arg"][arg.name]["type"] == "uint8" &&
            typeof predictTensorMap[opIdx]["arg"][arg.name]["value"] != "undefined" &&
            predictTensorMap[opIdx]["arg"][arg.name]["value"] == "128") {
          predictTensorMap[opIdx]["arg"][arg.name]["type"] = "int8";
          let tmpArray = [];
          for ( let val of Object.values(predictTensorMap[opIdx]["arg"][arg.name]["value"])) {
            tmpArray.push(val - 128);
          }
          predictTensorMap[opIdx]["arg"][arg.name]["value"] = tmpArray;
        }

        // NCHW => NHWC
        if (this._predictDataFormat && arg.name != "order" &&
            typeof predictTensorMap[opIdx]["arg"][arg.name]["type"] != "undefined" &&
            typeof predictTensorMap[opIdx]["arg"][arg.name]["value"] != "undefined") {
          if (predictTensorMap[opIdx]["arg"][arg.name]["value"].length == 4) {
            predictTensorMap[opIdx]["arg"][arg.name] = this._NCHWtoNHWCforArg(predictTensorMap[opIdx]["arg"][arg.name], true);
          } else {
            predictTensorMap[opIdx]["arg"][arg.name] = this._NCHWtoNHWCforArg(predictTensorMap[opIdx]["arg"][arg.name], false);
          }
        }
      }
    }

    return predictTensorMap;
  }

  _checkDataFormat() {
    this._checkInitDataFormat();
    this._checkpredictDataFormat();
    // console.log("initDataFormat: " + this._initDataFormat);
    // console.log("_predictDataFormat: " + this._predictDataFormat);
  }

  _checkInitDataFormat() {
    for (let op of this._init.op) {
      for (let arg of op.arg) {
        if (arg.name == "order") {
          let data = this._checkArgData(arg);
          let orderTmp = [];
          for (let val of data.value) {
            orderTmp.push(String.fromCharCode(val));
          }
          let formatStr = orderTmp.join("");
          if (formatStr == "NCHW") {
            this._initDataFormat = true;
            return
          };
        }
      }
    }
  }

  _checkpredictDataFormat() {
    for (let op of this._predict.op) {
      for (let arg of op.arg) {
        if (arg.name == "order") {
          let data = this._checkArgData(arg);
          let orderTmp = [];
          for (let val of data.value) {
            orderTmp.push(String.fromCharCode(val));
          }
          let formatStr = orderTmp.join("");
          if (formatStr == "NCHW") {
            this._predictDataFormat = true;
            return
          };
        }
      }
    }
  }

  _checkArgData(arg) {
    for (let [key, val] of Object.entries(arg)) {
      if (key != "name" && key != "tensors" && key != "nets" && key != "qtensors") {
        if (val.length !== 0) {
          return this._pareData(val, key);
        }
      }
    }
  }

  _pareData(dataValue, dataType) {
    switch(dataType) {
      case "i":
      case "ints": {
        if (dataValue.unsigned) {
          dataType = "uint32";
        } else {
          dataType = "int32";
        }

        let dataTmp = [];
        if (typeof dataValue.length == "undefined") {
          let value = dataValue.low;
          dataTmp.push(value);
        } else {
          for (let value of Object.values(dataValue)) {
            let valueNew = value.low;
            dataTmp.push(valueNew);
          };
        }
        dataValue = dataTmp;
      } break;
      case "f":
      case "floats": {
        dataType = "float32";
      } break;
      case "s": {
        dataType = "uint8";
        let dataTmp = [];
        let buf = Buffer.from(dataValue);
        for (let value of buf.values()) {
          dataTmp.push(value);
        }
        dataValue = dataTmp;
      } break;
      default: {
        throw new Error(`${dataType} is not supported.`);
      }
    };

    return {"type": dataType, "value": dataValue};
  }

  _NCHWtoNHWCforArg (arg, flag) {
    // For value
    let argValue = arg["value"];
    let argType = arg["type"];
    let argCtor = this._TypetoArray(argType);

    let N, C, H, W, tmpArgValue;
    if (flag) {
      N = arg["value"][0];
      C = arg["value"][1];
      H = arg["value"][2];
      W = arg["value"][3];
      tmpArgValue = new argCtor([N, H, W, C]);
    } else {
      tmpArgValue = new argCtor(this._DatatoArray(argValue));
    }

    arg["value"] = tmpArgValue;

    return arg;
  }

  _NCHWtoNHWCforTensor (tensor, flag) {
    // For shape
    let dataShape = tensor["shape"]["value"];
    let typeShape = tensor["shape"]["type"];
    let ctorShape = this._TypetoArray(typeShape);

    let N, C, H, W, tmpShapeValue;
    if (flag) {
      N = tensor["shape"]["value"][0];
      C = tensor["shape"]["value"][1];
      H = tensor["shape"]["value"][2];
      W = tensor["shape"]["value"][3];
      tmpShapeValue = new ctorShape([N, H, W, C]);
    } else {
      tmpShapeValue = new ctorShape(this._DatatoArray(dataShape));
    }

    tensor["shape"]["value"] = tmpShapeValue;

    // For value
    let dataValue = tensor["values"]["value"];
    let typeValue = tensor["values"]["type"];
    let ctorValue = this._TypetoArray(typeValue);

    let tmpDataValue;
    if (flag) {
      tmpDataValue = new ctorValue(dataValue.length);
      for (let n = 0; n < N; ++n) {
        for (let c = 0; c < C; ++c) {
          for (let h = 0; h < H; ++h) {
            for (let w = 0; w < W; ++w) {
              tmpDataValue[n*H*W*C + h*W*C + w*C + c] = dataValue[n*C*H*W + c*H*W + h*W + w];
            }
          }
        }
      }
    } else {
      tmpDataValue = new ctorValue(this._DatatoArray(dataValue));
    }

    tensor["values"]["value"] = tmpDataValue;

    // For zero_point
    if (typeof tensor["Y_zero_point"] != "undefined") {
      let dataPoint = tensor["Y_zero_point"]["value"];
      let typePoint = tensor["Y_zero_point"]["type"];
      let ctorPoint = this._TypetoArray(typePoint);
      let nhwcDataPoint = new ctorPoint(this._DatatoArray(dataPoint));
      tensor["Y_zero_point"]["value"] = nhwcDataPoint;
    }

    // For scale
    if (typeof tensor["Y_scales"] != "undefined") {
      let dataScale = tensor["Y_scales"]["value"];
      let typeScale = tensor["Y_scales"]["type"];
      let ctorScale = this._TypetoArray(typeScale);
      let nhwcDataScale = new ctorScale(this._DatatoArray(dataScale));
      tensor["Y_scales"]["value"] = nhwcDataScale;
    }

    return tensor;
  }

  _TypetoArray (type) {
    let ctor;
    if (type == "int32") ctor = Int32Array;
    else if (type == "uint32") ctor = Uint32Array;
    else if (type == "float32") ctor = Float32Array;
    else if (type == "uint8") ctor = Uint8Array;
    else if (type == "int8") ctor = Int8Array;
    else if (type == "str") ctor = Array;
    else throw new Error(`${type} is not supported.`);
    return ctor;
  }

  _DatatoArray (data) {
    let dataTmp = [];
    if (typeof data.length == "undefined" || data.length == 1) {
      dataTmp.push(data);
    } else {
      dataTmp = data;
    }
    return dataTmp;
  }
}

module.exports = Caffe2ModelUtils;
