class Caffe2ModelImporter {
  constructor(kwargs) {
    this._isQuantized = kwargs.isQuantized;
    this._netModel = kwargs.rawModel;
    this._weightModel = kwargs.weightModel;
    this._model = null;
    this._compilation = null;
    this._execution = null;
    this._tensorIds = [];
    this._tensorTypes = [];
    this._operations = [];
    this._operands = [];
    this._weightTensor = [...initCaffe2WeightTensor(this._weightModel)];
    this._requiredOps = new Set();
    this._options = {
      softmax: kwargs.softmax,
    };
    this._operandIndex = 0;
    this._backend = kwargs.backend;
    this._prefer = kwargs.prefer;
    this._inputScaleFactor = kwargs.inputScaleFactor;
    if (this._backend === 'WebML') {
      if (nnNative === null) {
        throw Error('Fails to initialize neural network context');
      }
      this._nn = nnNative;
    } else if (this._backend === 'WASM' || this._backend === 'WebGL') {
      this._nn = nnPolyfill;
    }
    this._bEagerMode = false;
    this._supportedOps = new Set();
  }

  setEagerMode = (flag) => {
    this._bEagerMode = flag;
  };

  setSupportedOps = (ops) => {
    this._supportedOps = ops;
  };

  async createCompiledModel() {
    let options = {
      backend: this._backend,
      eager: this._bEagerMode,
      supportedOps: this._supportedOps,
    };
    this._model = await this._nn.createModel(options);

    this._addTensorOperands();
    this._addOpsAndParams(); // Realization
    this._addInputsOutputs(); // Realization

    await this._model.finish();
    this._compilation = await this._model.createCompilation();

    let start = performance.now();
    this._compilation.setPreference(getPreferCode(this._backend, this._prefer));
    await this._compilation.finish();
    this._execution = await this._compilation.createExecution();
    let elapsed = performance.now() - start;
    console.log(`compilation time: ${elapsed.toFixed(2)} ms`);
  }

  async compute(inputTensors, outputTensors) {
    inputTensors.forEach((inputTensor, i) => {
      this._execution.setInput(i, inputTensor);
    });
    outputTensors.forEach((outputTensor, i) => {
      this._execution.setOutput(i, outputTensor);
    });

    let error = await this._execution.startCompute();
    if (error) {
      return error;
    }
    return 'success';
  }

  async * layerIterator(inputTensors, layerList) {
    const graph = this._netModel;

    const getLayerOutput = async (lastNodeIdx) => {
      this._tensorIds = [];
      this._tensorTypes = [];
      this._operations = [];
      this._operands = [];
      this._operandIndex = 0;
      if (this._backend !== 'WebML' && this._compilation) {
        this._compilation._preparedModel._deleteAll();
      }

      this._model = await this._nn.createModel({backend: this._backend});
      this._addTensorOperands();
      this._addOpsAndParams(); // Realization

      const lastNode = graph.op[lastNodeIdx];
      const output = lastNode.output[0];
      const input = lastNode.input;
      const inputIds = this._getTensorsId(input); // Realization
      const outputIds = this._getTensorsId(output); // Realization
      this._model.identifyInputsAndOutputs(inputIds, outputIds);

      await this._model.finish();
      this._compilation = await this._model.createCompilation();
      this._compilation.setPreference(getPreferCode(this._backend, this._prefer));
      await this._compilation.finish();
      this._execution = await this._compilation.createExecution();

      const outputSize = output.shape().reduce((a, b) => a * b); // Realization
      let outputTensor;
      if (this._isQuantized) {
        outputTensor = new Uint8Array(outputSize);
      } else {
        outputTensor = new Float32Array(outputSize);
      }
      await this.compute(inputTensors, [outputTensor]);
      return {
        layerId: lastNodeIdx,
        outputName: lastNode.name,
        tensor: outputTensor,
        outputIds: outputIds,
        inputIds: inputIds
      };
    };

    const operatorsLength = graph.op.length;
    if (typeof layerList === 'undefined') {
      for (let lastNode = 0; lastNode < operatorsLength;) {
        const layerOutput = await getLayerOutput(lastNode);
        yield layerOutput;
        lastNode = layerOutput.layerId + 1;
      }
    } else {
      for (let layerId of layerList) {
        if (layerId >= operatorsLength || layerId < 0) {
          throw new Error(`Illegal layer ${layerId}`);
        }
        yield await getLayerOutput(layerId);
      }
    }
  }

  // Add tensor
  _addTensorOperands() {
    let graph = this._netModel;

    for (let inputIdx in graph.externalInput) {
      this._addTensorByName(graph.externalInput[inputIdx]);
    }

    for (let outputIde in graph.externalOutput) {
      this._addTensorByName(graph.externalOutput[outputIde]);
    }
  }

  _addTensorByName(TensorName) {
    if (this._tensorIds[TensorName])
      throw new Error(`Tensor ${TensorName} is already added`);


  }
}
