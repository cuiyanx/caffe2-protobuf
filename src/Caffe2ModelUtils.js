function initCaffe2WeightTensor (weightModel) {
  let WeightTensor = new Map();

  for (let op of weightModel.op) {
    WeightTensor[op.output] = new Map();

    for (let arg of op.arg) {
      WeightTensor[op.output][arg.name] = new Map();
      let data = checkCaffe2WeightTensorData(arg);
      WeightTensor[op.output][arg.name]["type"] = data.type;
      WeightTensor[op.output][arg.name]["value"] = data.value;
    }
  }

  return WeightTensor;
}

function checkCaffe2WeightTensorData(arg) {
  for (let [key, val] of Object.entries(arg)) {
    if (key != "name" && key != "tensors" && key != "nets" && key != "qtensors") {
      if (val.length !== 0) {
        return pareCaffe2WeightTensorData(val, key);
      }
    }
  }
}

function pareCaffe2WeightTensorData(dataValue, dataType) {
  switch(dataType) {
    case "i":
    case "ints": {
      console.warn(`Tensor ${tensor.name} has Int64 data. Cast to a Int32 array.`);

      if (dataValue.unsigned) {
        dataType = "int32";
      } else {
        dataType = "uint32";
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
      console.info(`Tensor ${tensor.name} has float64 data. Cast to a float32 array.`);
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
