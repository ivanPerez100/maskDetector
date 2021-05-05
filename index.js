const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById("webcamButton");

function getUserMediaSupported(){
    return !!(navigator.mediaDevices && 
        navigator.mediaDevices.getUserMedia)
}

if (getUserMediaSupported()){
    enableWebcamButton.addEventListener('click', enableCam);
}else{
    console.warn('getUserMedia() is not supported by your browser');
}

function enableCam(event){
    if(!model){
        return;
    }

    event.target.classList.add('removed');

    const constraints = {
        video: true
    };

    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        video.srcObject = stream;
        video.addEventListener('loadeddata', predictWebcam);
    });
}

var children = [];

function predictWebcam(){
    return true;
}

const model_url = './maskDetector_tfjs/model.json';

var model = undefined;
// model =  await tf.loadGraphModel('./maskDetector_tfjs/model.json', 
// {onProgress: p => console.log(p)})
console.log("loading model");
await tf.loadGraphModel(model_url, 
{onProgress: p => console.log(p)}).then( function (loadedModel) {
    console.log("loaded model")
    model = loadedModel;
    demosSection.classList.remove('invisible');
});
console.log("done");