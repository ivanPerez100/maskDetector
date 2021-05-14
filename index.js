const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById("webcamButton");
const loadingMessage = document.getElementById("progress");
const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
const vh = Math.max(document.documentElement.clientHeight|| 0, window.innerHeight || 0);
var vidWidth = 0;
var vidHeight = 0;
var xStart = 0;
var yStart = 0;

function getUserMediaSupported(){
    return !!(navigator.mediaDevices && 
        navigator.mediaDevices.getUserMedia({video : true}))
}

async function getDevices(){
    const devices =  await navigator.mediaDevices.enumerateDevices();
    alert(devices);
    return devices;
}

if (getUserMediaSupported()){
    enableWebcamButton.addEventListener('click', enableCam);
    getDevices();
}else{
    console.warn('getUserMedia() is not supported by your browser');
    alert("Camera not supported");
}

function enableCam(event){
    if(!model){
        alert("Wait until the Model is fully loaded and for button to no loger be opaque!");
        return;
    }

    event.target.classList.add('removed');
    loadingMessage.classList.add('removed');

    const constraints = {
        video: true
    };

    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        let $video = document.querySelector('video');
        $video.srcObject = stream;
        $video.onloadedmetadata = () => {
            vidHeight = $video.clientHeight;
            vidWidth = $video.clientWidth;
            const imageSize = vidHeight * vidWidth;
            xStart = Math.floor((vw - vidWidth) / 10);
            yStart = (Math.floor( (vh - vidHeight) / 8) )>=0 ? (Math.floor((vh - vidHeight) / 8)):0;
            $video.play();
            $video.addEventListener('loadeddata', predictWebcam);
        }
    });
}


var children = [];

function predictWebcam(){
    detect(video).then(function () {
        window.requestAnimationFrame(predictWebcam);
    });
}

var classProbTreshold = 40;

async function detect(imgToPredict){
    tf.nextFrame();
    const tfimg = tf.browser.fromPixels(imgToPredict);
    const smallImg = tf.image.resizeBilinear(tfimg, [vidHeight, vidWidth]);
    const resized = tf.cast(smallImg, 'int32');
    var tf4d_ = tf.tensor4d(Array.from(resized.dataSync()), [1, vidHeight, vidWidth, 3]);
    const tf4d = tf.cast(tf4d_, 'int32');
    let predictions = await model.executeAsync(tf4d);
    drawPredictionBoxes(predictions[4].dataSync(), 
    predictions[7].dataSync(), 
    predictions[2].dataSync());
    tfimg.dispose();
    smallImg.dispose();
    resized.dispose();
    tf4d.dispose();
}

let classLabels = ["with_mask" ,"without_mask", "mask_worn_incorrect"];

function drawPredictionBoxes(predictionBoxes, predictionClasses, predictionScores){
    for( let i = 0; i< children.length; i++){
        liveView.removeChild(children[i]);
    }

    // vidHeight = video.clientHeight;
    // vidWidth = video.clientWidth;

    children.splice(0);
    for( let i= 0; i < predictionScores.length; i++){
        const minY = (predictionBoxes[i * 4] * vidHeight - yStart).toFixed(0);
        const minX = (predictionBoxes[i * 4 + 1] * vidWidth + video.offsetLeft).toFixed(0);
        const maxY = (predictionBoxes[i * 4 + 2] * vidHeight).toFixed(0);
        const maxX = (predictionBoxes[i * 4 + 3] * vidWidth + video.offsetLeft).toFixed(0);
        const score = predictionScores[i * 3] * 100;
        const width_ = (maxX-minX).toFixed(0);
        const height_ = (maxY-minY).toFixed(0);
        if (score > classProbTreshold && score < 100){
            const p = document.createElement('p');
            p.innerText = classLabels[(predictionClasses[i]) - 1] + " - with " +
            Math.round(score) + " % confidence";
            p.style = 'margin-left: ' + (minX) + 'px; margin-top: ' +
            (minY - 10) + "px; width: " +
            (width_ - 10) + "px; top: 0; left: 0;";

            const highlighter = document.createElement('div');
            highlighter.setAttribute('class', 'highlighter');
            highlighter.style = 'left: ' + minX + 'px; ' +
                'top: ' + minY + 'px; ' +
                'width: ' + (width_) + 'px; ' +
                'height: ' + (height_) + 'px;';
            liveView.appendChild(highlighter);
            liveView.appendChild(p);
            children.push(highlighter);
            children.push(p);
            
        }
    }
}

const model_url = './maskDetector_tfjs/model.json';
var model = undefined;

console.log("loading model");

await tf.loadGraphModel(model_url, 
    {onProgress: p => loadingMessage.innerHTML = 
        (p * 100) + '% of Model Loaded'}).then( function(loadedModel){
            console.log("done");
            model = loadedModel;
            webcamButton.classList.remove('invisible');
        }

);