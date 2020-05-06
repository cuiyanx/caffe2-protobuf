var protobuf = require("protobufjs");
var Long = require("long");
var caffe2 = require("./caffe2-local.js");
var fs = require('fs');
var init_path = "./models/init_net_int8.pb";
var predict_path = "./models/predict_net_int8.pb";
var JSONpath = "./output/caffe2-net.json";

var init_buffer = fs.readFileSync(init_path);
var predict_buffer = fs.readFileSync(predict_path);
var init_message = caffe2.caffe2.NetDef.decode(init_buffer);
var predict_message = caffe2.caffe2.NetDef.decode(predict_buffer);

var layers = new Map();
var externalInputs = new Map();

function pareData(dataValue, dataType) {
  switch(dataType) {
    case "i":
    case "ints": {
      if (dataValue.unsigned) {
        dataType = "int32";
      } else {
        dataType = "uint32";
      }
      let dataTmp = [];
      if (typeof dataValue.length == "undefined") {
        let value = new Long(dataValue.low, dataValue.high, dataValue.unsigned).toNumber();
        dataTmp.push(value);
      } else {
        for (let value of Object.values(dataValue)) {
          let valueNew = new Long(value.low, value.high, value.unsigned).toNumber();
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

function checkData(map) {
  for (let [key, val] of Object.entries(map)) {
    if (key != "name" && key != "tensors" && key != "nets" && key != "qtensors") {
      if (val.length !== 0) {
        return pareData(val, key);
      }
    }
  }
}

// externalInput
for (let inputName of predict_message.externalInput) {
  externalInputs[inputName] = new Map();

  for (let op of init_message.op) {
    if (op.output == inputName) {
      for (let arg of op.arg) {
        externalInputs[inputName][arg.name] = new Map();
        let data = checkData(arg);
        externalInputs[inputName][arg.name]["type"] = data.type;
        externalInputs[inputName][arg.name]["value"] = data.value;
      }
    }
  }
}

for (let opIdx in predict_message.op) {
  layers["layer-" + opIdx] = new Map();
  let op = predict_message.op[opIdx];

  // name, type, engine
  layers["layer-" + opIdx]["name"] = op.name;
  layers["layer-" + opIdx]["type"] = op.type;
  layers["layer-" + opIdx]["engine"] = op.engine;

  // input
  layers["layer-" + opIdx]["input"] = new Map();
  for (let inputIdx in op.input) {
    let input = op.input[inputIdx];

    if (inputIdx == 0) {
      layers["layer-" + opIdx]["input"][input] = "Up";
    } else {
      layers["layer-" + opIdx]["input"][input] = externalInputs[input];
    }
  }

  // output
  layers["layer-" + opIdx]["output"] = new Map();
  for (let outputIdx in op.output) {
    let output = op.output[outputIdx];

    layers["layer-" + opIdx]["output"][output] = "Down";
  }

  // arg
  layers["layer-" + opIdx]["arg"] = new Map();
  for (let argIdx in op.arg) {
    let arg = op.arg[argIdx];

    layers["layer-" + opIdx]["arg"][arg.name] = new Map();
    let data = checkData(arg);
    layers["layer-" + opIdx]["arg"][arg.name]["type"] = data.type;
    layers["layer-" + opIdx]["arg"][arg.name]["value"] = data.value;
  }
}

console.log(layers);
console.log(externalInputs);
console.log(predict_message);

fs.writeFileSync(JSONpath, JSON.stringify(layers, null, 4));
