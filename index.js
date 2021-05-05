const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById("webcamButton");
const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
const vh = Math.max(document.documentElement.clientHeight|| 0, window.innerHeight || 0);
var vidWidth = 0;
var vidHeight = 0;
var xStart = 0;
var yStart = 0;


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
        let $video = document.querySelector('video');
        $video.srcObject = stream;
        $video.onloadedmetadata = () => {
            vidHeight = $video.videoHeight;
            vidWidth = $video.videoWidth;
            xStart = Math.floor((vw - vidWidth) / 2);
            yStart = (Math.floor( (vh - vidHeight) / 2) )>=0 ? (Math.floor((vh - vidHeight) / 2)):0;
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

const imageSize = 520;
var classProbTreshold = 50;

async function detect(imgToPredict){
    tf.nextFrame();
    const tfimg = tf.browser.fromPixels(imgToPredict);
    const smallImg = tf.image.resizeBilinear(tfimg, [vidHeight, vidWidth]);
    const resized = tf.cast(smallImg, 'int32');
    var tf4d_ = tf.tensor4d(Array.from(resized.dataSync()), [1, vidHeight, vidWidth, 3]);
    const tf4d = tf.cast(tf4d_, 'int32');
    let predictions = await model.executeAsync(tf4d);

    // console.log(predictions[7].dataSync())
    // predictions.forEach(t => t.print());

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
    children.splice(0);
    // console.log(predictionClasses);
    for( let i= 0; i < 99; i++){
        const minY = (predictionBoxes[i * 4] * yStart+vidHeight).toFixed(0);
        const minX = (predictionBoxes[i * 4 + 1] * xStart+vidWidth).toFixed(0);
        const maxY = (predictionBoxes[i * 4 + 2] * yStart+vidHeight).toFixed(0);
        const maxX = (predictionBoxes[i * 4 + 3] * xStart+vidWidth).toFixed(0);
        const score = predictionScores[i * 3] * 100;
        const width_ = (maxX-minX).toFixed(0);
        const height_ = (maxY-minY).toFixed(0);
        //If confidence is above 70%
        console.log(predictionClasses[i]);
        if (score > classProbTreshold && score < 100){
            const highlighter = document.createElement('div');
            highlighter.setAttribute('class', 'highlighter');
            highlighter.style = 'left: ' + minX + 'px; ' +
                'top: ' + minY + 'px; ' +
                'width: ' + width_ + 'px; ' +
                'height: ' + height_ + 'px;';
            highlighter.innerHTML = '<p>'+Math.round(score) + '% ' + classLabels[(predictionClasses[i]) - 1] +'</p>';
            liveView.appendChild(highlighter);
            children.push(highlighter);
        }
    }
}

const model_url = './maskDetector_tfjs/model.json';

var model = undefined;

console.log("loading model");

tf.loadGraphModel(model_url, 
{onProgress: p => console.log(p)}).then( function (loadedModel) {
    console.log("loaded model")
    model = loadedModel;
    demosSection.classList.remove('invisible');
    console.log("done");
});
