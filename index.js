/*
    File: index.js
    
        In this file, we are write all of the code that is going to consist of
    checking for camera support, loading in our model, making predictions 
    with said loaded model, and displaying our preditions on the video camera
    feed shown on webpage. 

        This will be done with a number of functions which would be explained
    with their respective method headers and comments below.
*/

// First, we need to get elements from our HTML code 
const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById("webcamButton");
const loadingMessage = document.getElementById("progress");

// Get size of our current webpage. Later used to calculate where our bounding 
// box should be positioned later on when making predictions
const vw = Math.max(document.documentElement.clientWidth || 0, 
    window.innerWidth || 0);
const vh = Math.max(document.documentElement.clientHeight|| 0, 
    window.innerHeight || 0);

// Initialize variables that we would use later for bounding box positioning
var vidWidth = 0;
var vidHeight = 0;
var xStart = 0;
var yStart = 0;
var children = [];

// Class Labels that are going to be shown in bouding box
let classLabels = ["with_mask" ,"without_mask", "mask_worn_incorrect"];

// Probability Threshold that we would allow a bounding box to be drawn
var classProbTreshold = 40;

/**
 * Function: getUserMediaSupported
 * 
 * Input: None
 * 
 * Description: 
 * - In this function, we are determing if our current device (i.e Laptop, 
 * desktop , tablet, phone, etc) has a supported camera in which we could take 
 * in a video stream and start detecting if there is a person with or without a
 * mask in the stream.
 * 
 * - Later use this to determine if use is allowed to enable webcam or not
 * 
 * @returns {boolean} Boolean value determining if mediaDevices are 
 * supported or not
 */
function getUserMediaSupported(){
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
}

/**
 * Functions getDevices
 * 
 * @param {object} devicesInfo List returned by builtin .enumerateDevices()
 * 
 * Description:
 * - In this function we are going get a list of media device options. This 
 * would allow a user to decide which device the want to use start detecting.
 * 
 * - Detected options would be presented in an option dropdown selection found
 * in html page
 * 
 * - This is mainly done for people who are going to access the webapp on their
 * mobile devices since they have the option of using either their front or 
 * rear faced camera. However, it should still work for other devices such as
 * a person who has a builtin webcam and an external webcam connected.
 * 
 * 
 * 
 */
async function getDevices(devicesInfo){
    const devices =  await navigator.mediaDevices.enumerateDevices();
    // return devices;
}


/**
 * Funciton: enableCam
 * 
 * @param {*} event Event listener which detects if button was clicked
 * 
 * Description:
 * - In this function we are going enable our webcam so that we can start seeing
 * a video feed. However, we only allow this once our model is fully loaded. If
 * the user tries to press the button when the model is not yet fully loaded, 
 * they would be prompted iwth a message that tells them to wait until the model
 * is loaded in.
 * 
 * - Once model is loaded in, we would remove the button and loading progress 
 * that way we only have the video stream showing.
 * 
 * - Once the video stream starts playing, we would send this video stream into
 * our model and start predicting.
 * 
 * @returns {} Leaves function if model is not yet loaded
 */
function enableCam(event){

    // Check if model is loaded in, if not, then prompt user to wait.
    if(!model){
        alert("Wait for Model to fully load and for button to not be opaque!");
        return;
    }

    // Remove button and progress message as they are no longer needed
    event.target.classList.add('removed');
    loadingMessage.classList.add('removed');

    // Add constraint that ask for only video input
    const constraints = {
        video: true
    };


    // Get device that follow our constraints, which in this case, it's whatever
    // device allows video.
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {

        // Get video and stream video onto there
        let $video = document.querySelector('video');
        $video.srcObject = stream;
        $video.onloadedmetadata = () => {

            // Calculate some values that would be used for positioning our
            // bounding box and labels
            vidHeight = $video.clientHeight;
            vidWidth = $video.clientWidth;
            const imageSize = vidHeight * vidWidth;
            xStart = Math.floor((vw - vidWidth) / 10);
            yStart = (Math.floor( (vh - vidHeight) / 8) )>=0 ? 
                (Math.floor((vh - vidHeight) / 8)):0;

            // Play video and start predicting
            $video.play();
            $video.addEventListener('loadeddata', predictWebcam);
        }
    });
}

/**
 * Function: predictWebcam
 * 
 * Input: None
 * 
 * Description: 
 * - This is a function that is going to get the video stream and using that 
 * stream, it's going to get the current frame in the stream and call the detect 
 * function which is our function that makes a prediction.
 * 
 * Returns: None
 */
function predictWebcam(){
    detect(video).then(function () {
        window.requestAnimationFrame(predictWebcam);
    });
}

/**
 * Function: detect
 * 
 * @param {object} imgToPredict 
 * 
 * Description:
 * - In this function, we are getting a frame from our webcam feed and then 
 * converting the image into a tensor so that we can then pass it into the model
 * and come up with predictions which we would then pass into our function
 * that is going to draw our prediction boxes.
 * 
 * Returns: None
 */
async function detect(imgToPredict){

    // Get the frame and convert it onto a tensor
    tf.nextFrame();
    const tfimg = tf.browser.fromPixels(imgToPredict);
    const smallImg = tf.image.resizeBilinear(tfimg, [vidHeight, vidWidth]);
    const resized = tf.cast(smallImg, 'int32');
    var tf4d_ = tf.tensor4d(Array.from(resized.dataSync()), 
        [1, vidHeight, vidWidth, 3]);
    const tf4d = tf.cast(tf4d_, 'int32');
    
    // Take converted tensor and pass into model to get a prediction
    let predictions = await model.executeAsync(tf4d);

    // Take prediction and get the boxes, classes, and scores so that we can 
    // draw the boxes along with the class label and score.
    drawPredictionBoxes(predictions[4].dataSync(), 
    predictions[7].dataSync(), 
    predictions[2].dataSync());

    // To Save up memory, we are going to delete some of the tensors that we 
    // are no longer using anymore
    tfimg.dispose();
    smallImg.dispose();
    resized.dispose();
    tf4d.dispose();
}


/**
 * Function: drawPredictionBoxes
 * 
 * @param {Array} predictionBoxes 
 * @param {Array} predictionClasses 
 * @param {Array} predictionScores 
 * 
 * Description:
 * - In this function, we are going to be drawing the bounding boxes along with
 * displaying the detected class and confidence score.
 * 
 * 
 * Returns: None
 * 
 */
function drawPredictionBoxes(predictionBoxes, predictionClasses, predictionScores){

    // Remove any bounding box drawn before from the previous frame
    for( let i = 0; i < children.length; i++){
        liveView.removeChild(children[i]);
    }

    // Make sure that we start editing children array from the start
    children.splice(0);

    // Go through each prediction
    for( let i= 0; i < predictionScores.length; i++){

        // Get the coordinates and measurements for our bounding box
        //  - Special thing we had to do was to move the x coordinate by adding
        //    the .offsetLeft of the video. This is because we centered our 
        //    video frame so we have to take that into account.
        const minY = (predictionBoxes[i * 4] * vidHeight - yStart).toFixed(0);
        const minX = (predictionBoxes[i * 4 + 1] * vidWidth + 
            video.offsetLeft).toFixed(0);
        const maxY = (predictionBoxes[i * 4 + 2] * vidHeight).toFixed(0);
        const maxX = (predictionBoxes[i * 4 + 3] * vidWidth + 
            video.offsetLeft).toFixed(0);
        const score = predictionScores[i * 3] * 100;
        const width_ = (maxX-minX).toFixed(0);
        const height_ = (maxY-minY).toFixed(0);

        // Draw a bounding box only if the score exceeds the threshold
        if (score > classProbTreshold && score < 100){

            // Get Class Label
            const classLabel = classLabels[(predictionClasses[i]) - 1];

            // Determine what color should be our boundary box depending on 
            // the class label.
            var color = ''
            if( classLabel == "with_mask"){
                color = "rgba(10, 190, 25, 0.836)";
            }else if( classLabel == "without_mask"){
                color = "rgba(236, 37, 2, 0.836)";
            }else{
                color = "rgba(236, 142, 2, 0.836)";
            }

            // Create element that would hold our label and confidence score
            const p = document.createElement('p');

            // Write label and confidence score onto created element
            p.innerText = classLabel + " - with " +
            Math.round(score) + " % confidence";
            p.style = 'margin-left: ' + (minX) + 'px; margin-top: ' +
            (minY - 10) + "px; width: " +
            (width_ - 10) + "px; top: 0; left: 0;" + 
            'background-color: ' + color + ';';

            // Create element that would hold our bounding box
            const highlighter = document.createElement('div');
            highlighter.setAttribute('class', 'highlighter');

            // Draw boundary box using calculated coordinates and measurements
            highlighter.style = 'left: ' + minX + 'px; ' +
                'top: ' + minY + 'px; ' +
                'width: ' + (width_) + 'px; ' +
                'height: ' + (height_) + 'px;' + 
                'border: 2px solid ' + color +';';

            // Append/Push the label and boundary boxes created
            liveView.appendChild(highlighter);
            liveView.appendChild(p);
            children.push(highlighter);
            children.push(p);
            
        }
    }
}


// Check if the the user has a media device supported. If the user doesn't
// have a supported device, then we would let them know.
if (getUserMediaSupported()){
    enableWebcamButton.addEventListener('click', enableCam);
}else{
    console.warn('getUserMedia() is not supported by your browser');
    alert("Camera not supported");
}

// Get the path of our model.
const model_url = './maskDetector_tfjs/model.json';
var model = undefined;


// Load in our model and by using the onProgress option for loadGraphModel we 
// also display the progress of how much the model has been loaded in.
console.log("Loading Model");
tf.loadGraphModel(model_url, 
    {onProgress: p => loadingMessage.innerHTML = 
        Math.round(p * 100) + '% of Model Loaded'}).then( function(loadedModel){
            console.log("Done Loading Model");
            
            // Save the model
            model = loadedModel;

            // Make the button no longer Opaque
            webcamButton.classList.remove('invisible');
        }

);