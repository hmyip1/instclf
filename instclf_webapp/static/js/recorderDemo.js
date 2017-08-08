// manually rewritten from CoffeeScript output
// (see dev-coffee branch for original source)

// navigator.getUserMedia shim
navigator.getUserMedia =
  navigator.getUserMedia ||
  navigator.webkitGetUserMedia ||
  navigator.mozGetUserMedia ||
  navigator.msGetUserMedia;

// URL shim
window.URL = window.URL || window.webkitURL;

// audio context + .createScriptProcessor shim
var audioContext = new AudioContext;
if (audioContext.createScriptProcessor == null)
  audioContext.createScriptProcessor = audioContext.createJavaScriptNode;

// elements (jQuery objects)
var $testToneLevel = $('#test-tone-level'),
    $microphone = $('#microphone'),
    $microphoneLevel = $('#microphone-level'),
    $timeLimit = $('#time-limit'),
    $encoding = $('input[name="encoding"]'),
    $encodingOption = $('#encoding-option'),
    $encodingProcess = $('input[name="encoding-process"]'),
    $reportInterval = $('#report-interval'),
    $bufferSize = $('#buffer-size'),
    $recording = $('#recording'),
    $timeDisplay = $('#time-display'),
    $record = $('#record'),
    $cancel = $('#cancel'),
    $dateTime = $('#date-time'),
    $recordingList = $('#recording-list'),
    $modalLoading = $('#modal-loading'),
    $modalProgress = $('#modal-progress'),
    $modalError = $('#modal-error');

// initialize input element states (required for reloading page on Firefox)
$testToneLevel.attr('disabled', false);
$testToneLevel.valueAsNumber = 0;
$microphone.attr('disabled', false);
$microphone.checked = false;
$microphoneLevel.attr('disabled', false);
$microphoneLevel.valueAsNumber = 0;
$timeLimit.attr('disabled', false);
$timeLimit.valueAsNumber = 3;
$encoding.attr('disabled', false);
$encoding.checked = true;
$encodingProcess.attr('disabled', false);
$encodingProcess.checked = true;
$reportInterval.attr('disabled', false);
$reportInterval.valueAsNumber = 1;
$bufferSize.attr('disabled', false);

/*
test tone (440Hz sine with 2Hz on/off beep)
-------------------------------------------
            ampMod    output
osc(sine)-----|>--------|>----->(testTone)
              ^         ^
              |(gain)   |(gain)
              |         |
lfo(square)---+        0.5
*/
var testTone = (function() {
  var osc = audioContext.createOscillator(),
      lfo = audioContext.createOscillator(),
      ampMod = audioContext.createGain(),
      output = audioContext.createGain();
  lfo.type = 'square';
  lfo.frequency.value = 2;
  osc.connect(ampMod);
  lfo.connect(ampMod.gain);
  output.gain.value = 0.5;
  ampMod.connect(output);
  osc.start();
  lfo.start();
  return output;
})();

/*
master diagram
--------------
              testToneLevel
(testTone)----------|>---------+
                               |
                               v
                            (mixer)---+--->(audioRecorder)...+
                               ^      |                      :
              microphoneLevel  |      |                      v
(microphone)--------|>---------+      +--------------->(destination)
*/
var testToneLevel = audioContext.createGain(),
    microphone = undefined,     // obtained by user click
    microphoneLevel = audioContext.createGain(),
    mixer = audioContext.createGain();
testTone.connect(testToneLevel);
testToneLevel.gain.value = 0;
testToneLevel.connect(mixer);
microphoneLevel.gain.value = 0;
microphoneLevel.connect(mixer);
mixer.connect(audioContext.destination);

// audio recorder object
var audioRecorder = new WebAudioRecorder(mixer, {
  workerDir: 'static/js/',
  onEncoderLoading: function(recorder, encoding) {
    $modalLoading
      .find('.modal-title')
      .html("Loading " + encoding.toUpperCase() + " encoder ...");
    $modalLoading.modal('show');
  },
  onEncoderLoaded: function() { $modalLoading.modal('hide'); }
});

// mixer levels
$testToneLevel.on('input', function() {
  var level = $testToneLevel[0].valueAsNumber / 100;
  testToneLevel.gain.value = level * level;
});

$microphoneLevel.on('input', function() {
  var level = $microphoneLevel[0].valueAsNumber / 100;
  microphoneLevel.gain.value = level * level;
});

// obtaining microphone input
$microphone.click(function() {
  if (microphone == null)
    navigator.getUserMedia({ audio: true },
      function(stream) {
        microphone = audioContext.createMediaStreamSource(stream);
        microphone.connect(microphoneLevel);
        $microphone.attr('disabled', true);
        $microphoneLevel.removeClass('hidden');
      },
      function(error) {
        $microphone[0].checked = false;
        audioRecorder.onError(audioRecorder, "Could not get audio input.");
      });
});

// recording time limit
function plural(n) { return n > 1 ? "s" : ""; }

$timeLimit.on('input', function() {
  var min = $timeLimit[0].valueAsNumber;
  $('#time-limit-text').html("" + min + " minute" + plural(min));
});

// encoding selector + encoding options
var OGG_QUALITY = [-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    OGG_KBPS = [45, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 500],
    MP3_BIT_RATE = [64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
    ENCODING_OPTION = {
      wav: {
        label: '',
        hidden: true,
        max: 1,
        text: function(val) { return ''; }
      },
      ogg: {
        label: 'Quality',
        hidden: false,
        max: OGG_QUALITY.length - 1,
        text: function(val) {
          return OGG_QUALITY[val].toFixed(1) +
                 " (~" + OGG_KBPS[val] + "kbps)";
        }
      },
      mp3: {
        label: 'Bit rate',
        hidden: false,
        max: MP3_BIT_RATE.length - 1,
        text: function(val) { return "" + MP3_BIT_RATE[val] + "kbps"; }
      }
    },
    optionValue = {
      wav: null,
      ogg: 6,
      mp3: 5
    };

$encoding.click(function(event) {
  var encoding = $(event.target).attr('encoding'),
      option = ENCODING_OPTION[encoding];
  audioRecorder.setEncoding(encoding);
  $('#encoding-option-label').html(option.label);
  $('#encoding-option-text').html(option.text(optionValue[encoding]));
  $encodingOption
    .toggleClass('hidden', option.hidden)
    .attr('max', option.max);
  $encodingOption[0].valueAsNumber = optionValue[encoding];
});

$encodingOption.on('input', function() {
  var encoding = audioRecorder.encoding,
      option = ENCODING_OPTION[encoding];
  optionValue[encoding] = $encodingOption[0].value;
  $('#encoding-option-text').html(option.text(optionValue[encoding]));
});

// encoding process selector
var encodingProcess = 'background';

$encodingProcess.click(function(event) {
  encodingProcess = $(event.target).attr('mode');
  var hidden = encodingProcess === 'background';
  $('#report-interval-label').toggleClass('hidden', hidden);
  $reportInterval.toggleClass('hidden', hidden);
  $('#report-interval-text').toggleClass('hidden', hidden);
});

$reportInterval.on('input', function() {
  var sec = $reportInterval[0].valueAsNumber;
  $('#report-interval-text').html("" + sec + " second" + (plural(sec)));
});

// processor buffer size
var BUFFER_SIZE = [256, 512, 1024, 2048, 4096, 8192, 16384];

var defaultBufSz = (function() {
  var processor = audioContext.createScriptProcessor(undefined, 2, 2);
  return processor.bufferSize;
})();

var iDefBufSz = BUFFER_SIZE.indexOf(defaultBufSz);

$bufferSize[0].valueAsNumber = iDefBufSz;   // initialize with browser default

function updateBufferSizeText() {
  var iBufSz = $bufferSize[0].valueAsNumber,
      text = "" + BUFFER_SIZE[iBufSz];
  if (iBufSz === iDefBufSz)
    text += ' (browser default)';
  $('#buffer-size-text').html(text);
}

updateBufferSizeText();         // initialize text

$bufferSize.on('input', function() { updateBufferSizeText(); });

// save/delete recording
function saveRecording(blob, encoding) {
  var time = new Date(),
      url = URL.createObjectURL(blob),
      html = "<p recording='" + url + "'>" +
             "<audio controls src='" + url + "'></audio> " +
             " (" + encoding.toUpperCase() + ") " +
             time +
             " <a class='btn btn-default' href='" + url +
             "' download='recording." +
             encoding +
             "'>Save...</a> " +
             "<button class='btn btn-danger' recording='" +
             url +
             "'>Delete</button>" +
             "</p>";
  $recordingList.prepend($(html));
}

$recordingList.on('click', 'button', function(event) {
  var url = $(event.target).attr('recording');
  $("p[recording='" + url + "']").remove();
  URL.revokeObjectURL(url);
});

// time indicator
function minSecStr(n) { return (n < 10 ? "0" : "") + n; };

function updateDateTime() {
  var sec = audioRecorder.recordingTime() | 0;
  $timeDisplay.html(minSecStr(sec / 60 | 0) + ":" + minSecStr(sec % 60));
  $dateTime.html((new Date).toString());
};

window.setInterval(updateDateTime, 200);

// encoding progress report modal
var progressComplete = false;

function setProgress(progress) {
  var percent = (progress * 100).toFixed(1) + "%";
  $modalProgress
    .find('.progress-bar')
    .attr('style', "width: " + percent + ";");
  $modalProgress
    .find('.text-center')
    .html(percent);
  progressComplete = progress === 1;
};

$modalProgress.on("hide.bs.modal", function() {
  if (!progressComplete)
    audioRecorder.cancelEncoding();
});

// record | stop | cancel buttons
function disableControlsOnRecord(disabled) {
  if (microphone == null)
    $microphone.attr('disabled', disabled);
  $timeLimit.attr('disabled', disabled);
  $encoding.attr('disabled', disabled);
  $encodingOption.attr('disabled', disabled);
  $encodingProcess.attr('disabled', disabled);
  $reportInterval.attr('disabled', disabled);
  $bufferSize.attr('disabled', disabled);
};

function startRecording() {
  $recording.removeClass('hidden');
  $record.html('STOP');
  $cancel.removeClass('hidden');
  disableControlsOnRecord(true);
  audioRecorder.setOptions({
    timeLimit: $timeLimit[0].valueAsNumber * 60,
    encodeAfterRecord: encodingProcess === 'separate',
    progressInterval: $reportInterval[0].valueAsNumber * 1000,
    ogg: { quality: OGG_QUALITY[optionValue.ogg] },
    mp3: { bitRate: MP3_BIT_RATE[optionValue.mp3] }
  });
  audioRecorder.startRecording();
  setProgress(0);
};

function stopRecording(finish) {
  $recording.addClass('hidden');
  $record.html('RECORD');
  $cancel.addClass('hidden');
  disableControlsOnRecord(false);
  if (finish) {
    audioRecorder.finishRecording();
    if (audioRecorder.options.encodeAfterRecord) {
      $modalProgress
        .find('.modal-title')
        .html("Encoding " + audioRecorder.encoding.toUpperCase());
      $modalProgress.modal('show');
    }
  } else
    audioRecorder.cancelRecording();
};

$record.click(function() {
  if (audioRecorder.isRecording())
    stopRecording(true);
  else
    startRecording();
});

$cancel.click(function() { stopRecording(false); });

// event handlers
audioRecorder.onTimeout = function(recorder) {
  stopRecording(true);
};

audioRecorder.onEncodingProgress = function(recorder, progress) {
  setProgress(progress);
};

audioRecorder.onComplete = function(recorder, blob) {
  if (recorder.options.encodeAfterRecord)
    $modalProgress.modal('hide');
  saveRecording(blob, recorder.encoding);
};

audioRecorder.onError = function(recorder, message) {
  $modalError
    .find('.alert')
    .html(message);
  $modalError.modal('show');
};
