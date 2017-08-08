"use strict";

var root = (typeof self === 'object' && self.self === self && self) || (typeof global === 'object' && global.global === global && global) || this;

(function( global ) {

  var Recorder = function( config ){

    var that = this;

    if ( !Recorder.isRecordingSupported() ) {
      throw new Error("Recording is not supported in this browser");
    }

    this.state = "inactive";
    this.eventTarget = global.document.createDocumentFragment();
    this.audioContext = new global.AudioContext();
    this.monitorNode = this.audioContext.createGain();

    this.config = config = config || {};
    this.config.command = "init";
    this.config.bufferLength = config.bufferLength || 4096;
    this.config.monitorGain = config.monitorGain || 0;
    this.config.numberOfChannels = config.numberOfChannels || 1;
    this.config.originalSampleRate = this.audioContext.sampleRate;
    this.config.encoderSampleRate = config.encoderSampleRate || 48000;
    this.config.encoderPath = config.encoderPath || 'encoderWorker.min.js';
    this.config.streamPages = config.streamPages || false;
    this.config.leaveStreamOpen = config.leaveStreamOpen || false;
    this.config.maxBuffersPerPage = config.maxBuffersPerPage || 40;
    this.config.encoderApplication = config.encoderApplication || 2049;
    this.config.encoderFrameSize = config.encoderFrameSize || 20;
    this.config.resampleQuality = config.resampleQuality || 3;
    this.config.streamOptions = config.streamOptions || {
      optional: [],
      mandatory: {
        googEchoCancellation: false,
        googAutoGainControl: false,
        googNoiseSuppression: false,
        googHighpassFilter: false
      }
    };

    this.setMonitorGain( this.config.monitorGain );
    this.scriptProcessorNode = this.audioContext.createScriptProcessor( this.config.bufferLength, this.config.numberOfChannels, this.config.numberOfChannels );
    this.scriptProcessorNode.onaudioprocess = function( e ){
      that.encodeBuffers( e.inputBuffer );
    };
  };

  Recorder.isRecordingSupported = function(){
    return global.AudioContext && global.navigator && ( global.navigator.getUserMedia || ( global.navigator.mediaDevices && global.navigator.mediaDevices.getUserMedia ) );
  };

  Recorder.prototype.addEventListener = function( type, listener, useCapture ){
    this.eventTarget.addEventListener( type, listener, useCapture );
  };

  Recorder.prototype.clearStream = function() {
    if ( this.stream ) {

      if ( this.stream.getTracks ) {
        this.stream.getTracks().forEach(function ( track ) {
          track.stop();
        });
      }

      else {
        this.stream.stop();
      }

      delete this.stream;
    }
  };

  Recorder.prototype.encodeBuffers = function( inputBuffer ){
    if ( this.state === "recording" ) {
      var buffers = [];
      for ( var i = 0; i < inputBuffer.numberOfChannels; i++ ) {
        buffers[i] = inputBuffer.getChannelData(i);
      }

      this.encoder.postMessage({
        command: "encode",
        buffers: buffers
      });
    }
  };

  Recorder.prototype.initStream = function(){
    var that = this;

    var onStreamInit = function( stream ){
      that.stream = stream;
      that.sourceNode = that.audioContext.createMediaStreamSource( stream );
      that.sourceNode.connect( that.scriptProcessorNode );
      that.sourceNode.connect( that.monitorNode );
      that.eventTarget.dispatchEvent( new global.Event( "streamReady" ) );
      return stream;
    }

    var onStreamError = function( e ){
      that.eventTarget.dispatchEvent( new global.ErrorEvent( "streamError", { error: e } ) );
    }

    var constraints = { audio : this.config.streamOptions };

    if ( this.stream ) {
      this.eventTarget.dispatchEvent( new global.Event( "streamReady" ) );
      return global.Promise.resolve( this.stream );
    }

    if ( global.navigator.mediaDevices && global.navigator.mediaDevices.getUserMedia ) {
      return global.navigator.mediaDevices.getUserMedia( constraints ).then( onStreamInit, onStreamError );
    }

    if ( global.navigator.getUserMedia ) {
      return new global.Promise( function( resolve, reject ) {
        global.navigator.getUserMedia( constraints, resolve, reject );
      }).then( onStreamInit, onStreamError );
    }
  };

  Recorder.prototype.pause = function(){
    if ( this.state === "recording" ){
      this.state = "paused";
      this.eventTarget.dispatchEvent( new global.Event( 'pause' ) );
    }
  };

  Recorder.prototype.removeEventListener = function( type, listener, useCapture ){
    this.eventTarget.removeEventListener( type, listener, useCapture );
  };

  Recorder.prototype.resume = function() {
    if ( this.state === "paused" ) {
      this.state = "recording";
      this.eventTarget.dispatchEvent( new global.Event( 'resume' ) );
    }
  };

  Recorder.prototype.setMonitorGain = function( gain ){
    this.monitorNode.gain.value = gain;
  };

  Recorder.prototype.start = function(){
    if ( this.state === "inactive" && this.stream ) {
      var that = this;
      this.encoder = new global.Worker( this.config.encoderPath );

      if (this.config.streamPages){
        this.encoder.addEventListener( "message", function( e ) {
          that.streamPage( e.data );
        });
      }

      else {
        this.recordedPages = [];
        this.totalLength = 0;
        this.encoder.addEventListener( "message", function( e ) {
          that.storePage( e.data );
        });
      }

      // First buffer can contain old data. Don't encode it.
      this.encodeBuffers = function(){
        delete this.encodeBuffers;
      };

      this.state = "recording";
      this.monitorNode.connect( this.audioContext.destination );
      this.scriptProcessorNode.connect( this.audioContext.destination );
      this.eventTarget.dispatchEvent( new global.Event( 'start' ) );
      this.encoder.postMessage( this.config );
    }
  };

  Recorder.prototype.stop = function(){
    if ( this.state !== "inactive" ) {
      this.state = "inactive";
      this.monitorNode.disconnect();
      this.scriptProcessorNode.disconnect();

      if ( !this.config.leaveStreamOpen ) {
        this.clearStream();
      }

      this.encoder.postMessage({ command: "done" });
    }
  };

  Recorder.prototype.storePage = function( page ) {
    if ( page === null ) {
      var outputData = new Uint8Array( this.totalLength );
      var outputIndex = 0;

      for ( var i = 0; i < this.recordedPages.length; i++ ) {
        outputData.set( this.recordedPages[i], outputIndex );
        outputIndex += this.recordedPages[i].length;
      }

      this.eventTarget.dispatchEvent( new global.CustomEvent( 'dataAvailable', {
        detail: outputData
      }));

      this.recordedPages = [];
      this.eventTarget.dispatchEvent( new global.Event( 'stop' ) );
    }

    else {
      this.recordedPages.push( page );
      this.totalLength += page.length;
    }
  };

  Recorder.prototype.streamPage = function( page ) {
    if ( page === null ) {
      this.eventTarget.dispatchEvent( new global.Event( 'stop' ) );
    }

    else {
      this.eventTarget.dispatchEvent( new global.CustomEvent( 'dataAvailable', {
        detail: page
      }));
    }
  };


  // Exports
  global.Recorder = Recorder;

  if ( typeof define === 'function' && define.amd ) {
    define( [], function() {
      return Recorder;
    });
  }

  else if ( typeof module == 'object' && module.exports ) {
    module.exports = Recorder;
  }

})(root);

// (function(window){

//   var WORKER_PATH = 'static/recorderjs/recorderWorker.js';

//   var Recorder = function(source, cfg){
//     var config = cfg || {};
//     var bufferLen = config.bufferLen || 4096;
//     this.context = source.context;
//     if(!this.context.createScriptProcessor){
//        this.node = this.context.createJavaScriptNode(bufferLen, 2, 2);
//     } else {
//        this.node = this.context.createScriptProcessor(bufferLen, 2, 2);
//     }
   
//     var worker = new Worker(config.workerPath || WORKER_PATH);
//     worker.postMessage({
//       command: 'init',
//       config: {
//         sampleRate: this.context.sampleRate
//       }
//     });
//     var recording = false,
//       currCallback;

//     this.node.onaudioprocess = function(e){
//       if (!recording) return;
//       worker.postMessage({
//         command: 'record',
//         buffer: [
//           e.inputBuffer.getChannelData(0),
//           e.inputBuffer.getChannelData(1)
//         ]
//       });
//     }

//     this.configure = function(cfg){
//       for (var prop in cfg){
//         if (cfg.hasOwnProperty(prop)){
//           config[prop] = cfg[prop];
//         }
//       }
//     }

//     this.record = function(){
//       recording = true;
//     }

//     this.stop = function(){
//       recording = false;
//     }

//     this.clear = function(){
//       worker.postMessage({ command: 'clear' });
//     }

//     this.getBuffers = function(cb) {
//       currCallback = cb || config.callback;
//       worker.postMessage({ command: 'getBuffers' })
//     }

//     this.exportWAV = function(cb, type){
//       currCallback = cb || config.callback;
//       type = type || config.type || 'audio/wav';
//       if (!currCallback) throw new Error('Callback not set');
//       worker.postMessage({
//         command: 'exportWAV',
//         type: type
//       });
//     }

//     this.exportMonoWAV = function(cb, type){
//       currCallback = cb || config.callback;
//       type = type || config.type || 'audio/wav';
//       if (!currCallback) throw new Error('Callback not set');
//       worker.postMessage({
//         command: 'exportMonoWAV',
//         type: type
//       });
//     }

//     worker.onmessage = function(e){
//       var blob = e.data;
//       currCallback(blob);
//     }

//     source.connect(this.node);
//     this.node.connect(this.context.destination);   // if the script node is not connected to an output the "onaudioprocess" event is not triggered in chrome.
//   };

//   Recorder.setupDownload = function(blob, filename){
//     var url = (window.URL || window.webkitURL).createObjectURL(blob);
//     var link = document.getElementById("save");
//     link.href = url;
//     link.download = filename || 'output.wav';
//   }

//   window.Recorder = Recorder;

// })(window);
