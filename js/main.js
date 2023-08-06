// Add labels
const labels = [
  
  "bruise",
  "berry",

];

// React State implementation in Vanilla JS
const useState = (defaultValue) => {
  let value = defaultValue;
  const getValue = () => value;
  const setValue = (newValue) => (value = newValue);
  return [getValue, setValue];
};

// Declare letiables
const numClass = labels.length;
const [session, setSession] = useState(null);
let mySession;
// Configs
const modelName = "yolov8n-seg.onnx";
const modelInputShape = [1, 3, 640, 640];
const topk = 100;
const iouThreshold = 0.2;
const confThreshold = 0.85    ;
const classThreshold = 0.85;

// wait until opencv.js initialized
cv["onRuntimeInitialized"] = async () => {
  // create session
  const [yolov5, nms, mask, segnet, nmsv8] = await Promise.all([
    ort.InferenceSession.create(`model/best-yolov8.onnx`),
    ort.InferenceSession.create(`model/nms-yolov5.onnx`),
    ort.InferenceSession.create("model/mask-yolov8-seg.onnx"),
    ort.InferenceSession.create("model/best-seg-640-yolov8.onnx"),
    ort.InferenceSession.create(`model/nms-yolov8.onnx`),
  ]);

  // warmup main model
  const tensor = new ort.Tensor(
    "float32",
    new Float32Array(modelInputShape.reduce((a, b) => a * b)),
    modelInputShape
  );
  await yolov5.run({ images: tensor });
  mySession = setSession({ net: yolov5, nms: nms, mask: mask, segnet: segnet, nmsv8: nmsv8 });
};

// Detect Image Function
const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  confThreshold,
  classThreshold,
  inputShape
) => {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas
  var bruise_ratio_list = [];
  const [modelWidth, modelHeight] = inputShape.slice(2);
  const maxSize = Math.max(modelWidth, modelHeight);
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight);
//////console.log(input.data32F);
  const tensor = new ort.Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const config = new ort.Tensor(
    "float32",
    new Float32Array([2, topk, iouThreshold, confThreshold])
  ); // nms config tensor
  const { output0, output1 } = await session.net.run({ images: tensor });
  //const output0 = output;
  // run session and get output layer
    const { selected } = await session.nmsv8.run({ detection: output0, config: config }); 
  //const { selected } = await session.nmsv8.run({
  //  detection: output0,
  //  config: config,
  //}); // get selected idx from nms
////console.log(selected_idx);

  const boxes = []; // ready to draw boxes
  const overlay = cv.Mat.zeros(modelHeight, modelWidth, cv.CV_8UC4); // create overlay to draw segmentation object

  // looping through output
  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]); // get rows
    let box = data.slice(0, 4); // det boxes
    const scores = data.slice(4, 4 + numClass); // det classes probability scores
    const score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores
    const color = colors.get(label); // get color

    box = overflowBoxes(
      [
        box[0] - 0.5 * box[2], // before upscale x
        box[1] - 0.5 * box[3], // before upscale y
        box[2], // before upscale w
        box[3], // before upscale h
      ],
      maxSize
    ); // keep boxes in maxSize range

    const [x, y, w, h] = overflowBoxes(
      [
        Math.floor(box[0] * xRatio), // upscale left
        Math.floor(box[1] * yRatio), // upscale top
        Math.floor(box[2] * xRatio), // upscale width
        Math.floor(box[3] * yRatio), // upscale height
      ],
      maxSize
    ); // upscale boxes

    boxes.push({
      label: labels[label],
      probability: score,
      color: color,
      bounding: [x, y, w, h], // upscale box
    }); // update boxes to draw later
////console.log(x,y,w,h);

//      const mask = new ort.Tensor(
//        "float32",
//        new Float32Array([
//          ...box, // original scale box
//          ...data.slice(5 + numClass), // mask data
//        ])
//      ); // mask input
//      const maskConfig = new ort.Tensor(
//        "float32",
//        new Float32Array([
//          maxSize,
//          x, // upscale x
//          y, // upscale y
//          w, // upscale width
//          h, // upscale height
//          ...Colors.hexToRgba(color, 120), // color in RGBA
//        ])
//      ); // mask config
//      const { mask_filter } = await session.mask.run({
//        detection: mask,
//        mask: output1,
//        config: maskConfig,
//      }); // get mask
//
//      const mask_mat = cv.matFromArray(
//        mask_filter.dims[0],
//        mask_filter.dims[1],
//        cv.CV_8UC4,
//        mask_filter.data
//      ); // mask result to Mat
//
//      cv.addWeighted(overlay, 1, mask_mat, 1, 0, overlay); // Update mask overlay
//      mask_mat.delete(); // delete unused Mat
  //  }
  }

 //const mask_img = new ImageData(
 //  new Uint8ClampedArray(overlay.data),
 //  overlay.cols,
 //  overlay.rows
 //); // create image data from mask overlay
 //ctx.putImageData(mask_img, 0, 0); // put ImageData data to canvas

  renderBoxes(ctx, boxes); // Draw boxes
  //////console.log(imgdata)
  
//  var imgdata=ctx.getImageData(200,200, 100, 100); // doesnt get the image only gets the bounding boxes
//  ctx.putImageData(imgdata, 20, 20);

// Create a new canvas element for displaying the regionImageData
const regionCanvas = document.createElement('canvas');
////console.log(regionCanvas);
// Set the dimensions of the regionCanvas
regionCanvas.width = 640;
regionCanvas.height = 640;

// Get the 2D rendering context of the regionCanvas
const regionCtx = regionCanvas.getContext('2d');
regionCtx.clearRect(0,0, regionCanvas.width, regionCanvas.height);
// Draw the regionImageData onto the regionCanvas
regionCtx.drawImage(image, 0,0,image.naturalWidth, image.naturalHeight, 0, 0, 640, 640); //putImageData(regionImageData, 0, 0);
// 393 342 32 27

//regionCtx.clearRect(0, 0, regionCanvas.width, regionCanvas.height);

//regionCtx.drawImage(image, (393/canvas.width)*image.naturalWidth, (342/canvas.height)*image.naturalHeight, (32/canvas.width)*image.naturalWidth, (27/canvas.height)*image.naturalHeight, 0, 0, 200, 200);
//regionCtx.drawImage(image, 393, 342, 32, 27, 0, 0, 200, 200);
//// Create a new image element
const img = new Image();
//
//// Set the source of the image to a data URL representing the regionImageData
img.src = regionCanvas.toDataURL();



//////console.log(img);
//regionCtx.clearRect(0, 0, regionCanvas.width, regionCanvas.height);
////console.log(img.src);




const loadCrops = () => {
  return new Promise(resolve => {

//const regionCanvas1 = document.createElement('canvas');
//////////console.log(regionCanvas1);
//////////console.log(regionCanvas);
// Set the dimensions of the regionCanvas
//regionCanvas1.width = 800;
//regionCanvas1.height = 10000;
//////////console.log(boxes);
    

img.onload = function() {
	height_per_box = 40;
	niter = 0 ;
	boxes.forEach((box) => {
		

	
const [x1, y1, width, height] =box.bounding; //  [341, 198, 34, 27]; // 
//console.log(x1, y1, width, height);

// Create a new canvas element for displaying the regionImageData
const regionCanvas1 = document.createElement('canvas');
regionCanvas1.width = (width/640)*image.naturalWidth;
regionCanvas1.height =  (height/640)*image.naturalHeight;	
// Get the 2D rendering context of the regionCanvas
const regionCtx1 = regionCanvas1.getContext('2d');
//regionCtx1.drawImage(image, (x1/640)*image.naturalWidth, (y1/640)*image.naturalHeight, (width/640)*image.naturalWidth, (height/640)*image.naturalHeight, ((0+50)/640)*image.naturalWidth, niter*(height_per_box/640)*image.naturalHeight, (width/640)*image.naturalWidth, (height/640)*image.naturalHeight);
//regionCtx1.drawImage(image, (x1/640)*image.naturalWidth, (y1/640)*image.naturalHeight, (width/640)*image.naturalWidth, (height/640)*image.naturalHeight, 0,0, (width/640)*image.naturalWidth, (height/640)*image.naturalHeight);
regionCtx1.drawImage(image, (x1/640)*image.naturalWidth, (y1/640)*image.naturalHeight, (width/640)*image.naturalWidth, (height/640)*image.naturalHeight, 0,0, regionCanvas1.width, regionCanvas1.height);
bruise_ratio_list = [];
niter = niter+1;



var  img1 = new Image();
//// Set the source of the image to a data URL representing the regionImageData
img1.src = regionCanvas1.toDataURL();
//console.log(img1.src);

document.body.appendChild(regionCanvas1);

var  img2 = new Image();
img1.onload = async () => {
var bruise_pixels = 0;
var berry_pixels = 0;
  const iouThreshold = 0.7;
const confThreshold = 0.7;
const classThreshold = 0.2;
  topk = 100;
 const [input, xRatio, yRatio] = preprocessing(img1, modelWidth, modelHeight); // preprocess frame

  const tensor1 = new ort.Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const config = new ort.Tensor(
    "float32",
    new Float32Array([
      numClass, // num class
      topk, // topk per class
      iouThreshold, // iou threshold
      confThreshold, // score threshold
    ])
  ); // nms config tensor
  const { output0, output1 } = await session.segnet.run({ images: tensor1 }); // run session and get output layer. out1: detect layer, out2: seg layer
  const { selected } = await session.nmsv8.run({ detection: output0, config: config }); // perform nms and filter boxes

  const boxes1 = []; // ready to draw boxes
  const overlay = cv.Mat.zeros(modelHeight, modelWidth, cv.CV_8UC4); // create overlay to draw segmentation object

  // looping through output
  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]); // get rows
    let box1 = data.slice(0, 4); // det boxes
    const scores = data.slice(4, 4 + numClass); // det classes probability scores
    const score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores
    const color = colors.get(label); // get color

    box1 = overflowBoxes(
      [
        box1[0] - 0.5 * box1[2], // before upscale x
        box1[1] - 0.5 * box1[3], // before upscale y
        box1[2], // before upscale w
        box1[3], // before upscale h
      ],
      maxSize
    ); // keep boxes in maxSize range

    const [x, y, w, h] = overflowBoxes(
      [
        Math.floor(box1[0] * xRatio), // upscale left
        Math.floor(box1[1] * yRatio), // upscale top
        Math.floor(box1[2] * xRatio), // upscale width
        Math.floor(box1[3] * yRatio), // upscale height
      ],
      maxSize
    ); // upscale boxes

    boxes1.push({
      label: labels[label],
      probability: score,
      color: color,
      bounding: [x, y, w, h], // upscale box
    }); // update boxes to draw later

console.log(data.slice(4 + numClass));
    const mask = new ort.Tensor(
      "float32",
      new Float32Array([
        ...box1, // original scale box
        ...data.slice(4 + numClass), // mask data
      ])
    ); // mask input
    const maskConfig = new ort.Tensor(
      "float32",
      new Float32Array([
        maxSize,
        x, // upscale x
        y, // upscale y
        w, // upscale width
        h, // upscale height
        ...Colors.hexToRgba(color, 120), // color in RGBA
      ])
    ); // mask config
	
    const { mask_filter } = await session.mask.run({
      detection: mask,
      mask: output1,
      config: maskConfig,
    }); // perform post-process to get mask

    const mask_mat = cv.matFromArray(
      mask_filter.dims[0],
      mask_filter.dims[1],
      cv.CV_8UC4,
      mask_filter.data
    ); // mask result to Mat

    cv.addWeighted(overlay, 1, mask_mat, 1, 0, overlay); // add mask to overlay
    mask_mat.delete(); // delete unused Mat
  }

//const bruise_ratio = bruise_pixels / berry_pixels;

 const mask_img = new ImageData(
   new Uint8ClampedArray(overlay.data),
   overlay.cols,
   overlay.rows
 ); // create image data from mask overlay




 
 regionCtx1.globalCompositeOperation = 'source-over';
// regionCtx1.putImageData(mask_img,0,0);// 0, 0); // put ImageData data to canvas
// Create a temporary canvas for overlaying the image data
const tempCanvas = document.createElement('canvas');
tempCanvas.width = img1.naturalWidth; // regionCanvas1.width;
tempCanvas.height = img1.naturalHeight; //overlay.rowsregionCanvas1.height;
const tempCtx = tempCanvas.getContext('2d');


const tempCanvas1 = document.createElement('canvas');
tempCanvas1.width = overlay.cols; // regionCanvas1.width;
tempCanvas1.height = overlay.rows; //overlay.rowsregionCanvas1.height;
const tempCtx1 = tempCanvas1.getContext('2d');

// Draw the original image onto the temporary canvas
//tempCtx.drawImage(regionCanvas1, 0, 0);
// Draw the mask image data on top of the existing image on the temporary canvas
const opacity = 0.5; // Opacity value between 0 and 1
 tempCtx1.globalCompositeOperation = 'source-over';
 tempCtx1.globalAlpha = opacity;
 tempCtx1.putImageData(mask_img, 0, 0);
 // Get the ImageData object from the canvas


 
tempCtx.globalCompositeOperation = 'source-over';
 // Draw boxes
// Set the global alpha value to control the opacity
tempCtx.globalAlpha = opacity;
  
 tempCtx.globalCompositeOperation = 'source-over';
tempCtx.drawImage(img1, 0,0,img1.naturalWidth, img1.naturalHeight, 0,0,img1.naturalWidth, img1.naturalHeight );
 //renderBoxes(tempCtx, boxes1);
tempCtx.drawImage(tempCanvas1, 0,0,overlay.cols,overlay.rows, 0,0, img1.naturalWidth, img1.naturalHeight)
//console.log(ctx.canvas.width, ctx.canvas.height);
ctx.drawImage(tempCanvas1, 0,0,overlay.cols,overlay.rows, x1, y1, width, height);
//tempCtx.globalAlpha = 1.0;


//////////////// for ratio
const tempCanvas2 = document.createElement('canvas');
tempCanvas2.width = img1.naturalWidth; // regionCanvas1.width;
tempCanvas2.height = img1.naturalHeight; //overlay.rowsregionCanvas1.height;
const tempCtx2 = tempCanvas2.getContext('2d');
tempCtx2.drawImage(tempCanvas1, 0,0,overlay.cols,overlay.rows, 0,0, img1.naturalWidth, img1.naturalHeight)
const imageData2 = tempCtx2.getImageData(0, 0, tempCanvas2.width, tempCanvas2.height);
const colorFrequencyMap = countColors(imageData2);
//console.log("Updtaed Color Frequency Map:");
//console.log(colorFrequencyMap);



berry_pixels = countNonZeroPixels(imageData2);
bruise_pixels = countPixelsWithCondition(imageData2);
var bruise_ratio1 = bruise_pixels/berry_pixels;
bruise_ratio_list.push(bruise_ratio1);
document.body.appendChild(tempCanvas);

//  renderBoxes(regionCtx1, boxes1); // Draw boxes

 // regionCtx1.drawImage(tempCanvas,0,0,regionCanvas1.width,regionCanvas1.height, 0,0,regionCanvas1.width,regionCanvas1.height);
  ////////////////console.log(imgdata)
  document.body.appendChild(regionCanvas1);
 // document.body.appendChild(tempCanvas2);
  
  // Create a new paragraph element
const paragraph = document.createElement('p');
// Set the text content of the paragraph
paragraph.textContent = `Bruise Pixels: ${bruise_pixels}, Berry Pixels: ${berry_pixels}, Ratio: ${(bruise_ratio1*100).toFixed(1)}`;
// Append the paragraph to the container
document.body.appendChild(paragraph);

var lineBreak = document.createElement("br");
document.body.appendChild(lineBreak);

console.log(bruise_ratio_list.length, boxes.length);
if (bruise_ratio_list.length == boxes.length) 
{
  sum_ratio=bruise_ratio_list.reduce((previous, current) => current += previous);
  mean_ratio=sum_ratio/bruise_ratio_list.length;
  renderBoxes1(ctx, boxes,bruise_ratio_list ); 
  alert('Now here');
    // Create a new paragraph element
const paragraph1 = document.createElement('p');
// Set the text content of the paragraph
paragraph1.textContent = `Mean bruise ratio ${(mean_ratio*100).toFixed(1)}`;
// Append the paragraph to the container
document.body.appendChild(paragraph1);

console.log('Mean bruise ratio');
console.log(mean_ratio);
}

//  var imgdata=ctx.getImageData(200,200, 100, 100); // doesnt get the image only gets the bounding boxes
//  ctx.putImageData(imgdata, 20, 20);

overlay.delete();
input.delete(); 

};
img2.src = regionCanvas1.toDataURL();
//console.log(img2.src);
//document.body.appendChild(regionCanvas1);

//document.body.appendChild(img1);

////////////////console.log(img1.src);
	 });
	//regionCtx1.drawImage(img, 524, 343, 34, 29, 0, 0, regionCanvas1.width, regionCanvas1.height);
	//regionCtx1.drawImage(img, 341, 198, 34, 27, 0, 0, regionCanvas1.width, regionCanvas1.height);
  //regionCtx1.drawImage(img, 472, 435, 45, 39, 0, 0, regionCanvas1.width, regionCanvas1.height);
  //regionCtx1.drawImage(img, 393, 342, 32, 27, 0, 0, regionCanvas1.width, regionCanvas1.height);
 // //////////////console.log(regionCanvas1.toDataURL());
};

	
	
  });
};

const drawCrops = async () => {
	//  alert('Now here');
	await loadCrops();


};

drawCrops();


//regionCtx1.clearRect(0,0, regionCanvas1.width, regionCanvas1.height);

//regionCtx1.drawImage(img, 393, 342, 32, 27, 0, 0, 200, 200);
////////////////console.log( regionCanvas1.toDataURL());

// Set any additional attributes or styles for the image if desired
//img.style.width = '1000px' ; //regionImageData.width + 'px';
//img.style.height =  '1000px'; //regionImageData.height + 'px';

// Append the image to the document body

//document.body.appendChild(regionCanvas);	
//document.body.appendChild(img);	 

	  
	  
  input.delete(); // delete unused Mat
  overlay.delete(); // delete unused Mat
};

/**
 * Get divisible image size by stride
 * @param {Number} stride
 * @param {Number} width
 * @param {Number} height
 * @returns {Number[2]} image size [w, h]
 */
const divStride = (stride, width, height) => {
  if (width % stride !== 0) {
    if (width % stride >= stride / 2)
      width = (Math.floor(width / stride) + 1) * stride;
    else width = Math.floor(width / stride) * stride;
  }
  if (height % stride !== 0) {
    if (height % stride >= stride / 2)
      height = (Math.floor(height / stride) + 1) * stride;
    else height = Math.floor(height / stride) * stride;
  }
  return [width, height];
};

/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @param {Number} stride model stride
 * @return preprocessed image and configs
 */
const preprocessing = (source, modelWidth, modelHeight, stride = 32) => {
  const mat = cv.imread(source); // read from img tag
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR

  const [w, h] = divStride(stride, matC3.cols, matC3.rows);
  cv.resize(matC3, matC3, new cv.Size(w, h));

  // padding image to [n x n] dim
  const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
  const xPad = maxSize - matC3.cols, // set xPadding
    xRatio = maxSize / matC3.cols; // set xRatio
  const yPad = maxSize - matC3.rows, // set yPadding
    yRatio = maxSize / matC3.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image
  cv.copyMakeBorder(
    matC3,
    matPad,
    0,
    yPad,
    0,
    xPad,
    cv.BORDER_CONSTANT,
    [0, 0, 0, 255]
  ); // padding black

  const input = cv.blobFromImage(
    matPad,
    1 / 255.0, // normalize
    new cv.Size(modelWidth, modelHeight), // resize to model input size
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  ); // preprocessing image matrix

  // release mat opencv
  mat.delete();
  matC3.delete();
  matPad.delete();

  return [input, xRatio, yRatio];
};

/**
 * Handle overflow boxes based on maxSize
 * @param {Number[4]} box box in [x, y, w, h] format
 * @param {Number} maxSize
 * @returns non overflow boxes
 */
const overflowBoxes = (box, maxSize) => {
  box[0] = box[0] >= 0 ? box[0] : 0;
  box[1] = box[1] >= 0 ? box[1] : 0;
  box[2] = box[0] + box[2] <= maxSize ? box[2] : maxSize - box[0];
  box[3] = box[1] + box[3] <= maxSize ? box[3] : maxSize - box[1];
  return box;
};

function countPixelsWithCondition(imageData) {
  const data = imageData.data;
  
  const count = Array.from(data).reduce((accumulator, currentValue, index) => {
    if (index % 4 === 0 && currentValue > 100 && data[index + 1] < 100 && data[index + 2] < 100) {
      return accumulator + 1;
    } else {
      return accumulator;
    }
  }, 0);
  
  return count;
}

function countNonZeroPixels(imageData) {
  const data = imageData.data;

  // Convert pixel data to an array of colors
  const colors = Array.from({ length: data.length / 4 }, (_, i) => {
    const startIndex = i * 4;
    const r = data[startIndex];
    const g = data[startIndex + 1];
    const b = data[startIndex + 2];
    return `${r},${g},${b}`;
  });

  // Filter out colors with RGB(0, 0, 0)
  const nonZeroColors = colors.filter(color => color !== "0,0,0");

  // Return the count of non-zero pixels
  return nonZeroColors.length;
}


function countColors(imageData) {
  const data = imageData.data;
  const colorMap = new Map();

  // Iterate over pixel data
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];
    const color = `${r},${g},${b},${a}`;

    // Update color frequency in the map
    if (colorMap.has(color)) {
      colorMap.set(color, colorMap.get(color) + 1);
    } else {
      colorMap.set(color, 1);
    }
  }

  return colorMap;
}
// Map(5) {'0,0,0' => 212291, '255,56,56' => 57644, '255,157,151' => 139100, '255,213,207' => 387, '255,112,112' => 178}



/////////////////////////////////// render boxes with bruise ratio


/* 
Render Boxes
*/
const renderBoxes1 = (ctx, boxes, blist) => {
  // font configs
  const font = `${Math.max(
    Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40),
    14
  )}px Arial`;

  var e = document.getElementById("fbbox");
  var fvalue = e.value;
  ctx.font = fvalue + 'px Arial' //font;
  ctx.textBaseline = "top";
   ind  = 0
  boxes.forEach((box) => {
	  bratio = blist[ind];
	  
	  ind = ind +1;
    ////////////////console.log(box);
    const className = bratio; //box.label;
	
	
	var e = document.getElementById("colorbbox");
	var cvalue = e.value;
    const color = cvalue; //box.color;
    const score = (bratio*100).toFixed(1); //(box.probability * 100).toFixed(1);
    const [x1, y1, width, height] = box.bounding;


	var e = document.getElementById("linewidthbbox");
	var lwvalue = e.value;


    // draw border box
    ctx.strokeStyle = color;
    ctx.lineWidth =  parseFloat(lwvalue); //1.0;//Math.max(

    //  Math.min(ctx.canvas.width, ctx.canvas.height) / 200,
    //  2.5
    //);
    ctx.strokeRect(x1, y1, width, height);
	 
	  if (document.getElementById('bruisebbox').checked) {
      // draw the label background.
     ctx.fillStyle = color;
     const textWidth = ctx.measureText(score + "%").width; //ctx.measureText(className + " - " + score + "%").width;
     const textHeight = parseInt(font, 10); // base 10
     const yText = y1 - (textHeight + ctx.lineWidth);
     ctx.fillRect(
       x1 - 1,
       yText < 0 ? 0 : yText,
       textWidth + ctx.lineWidth,
       textHeight + ctx.lineWidth
     );
	 
     // Draw labels
     ctx.fillStyle = "#ffffff";
     ctx.fillText(
       score + "%",
       x1 - 1,
       yText < 0 ? 1 : yText + 1
     );
	 }
 	 
  });
};

//////////////////////////////////////////////////////////////////


/* 
Render Boxes
*/
const renderBoxes = (ctx, boxes) => {
  // font configs
  const font = `${Math.max(
    Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40),
    14
  )}px Arial`;
  ctx.font = font;
  ctx.textBaseline = "top";

  boxes.forEach((box) => {
    ////////////////console.log(box);
    const className = box.label;
    	
	var e = document.getElementById("colorbbox");
	var cvalue = e.value;
    const color = cvalue; //box.color;
	
    const score = (box.probability * 100).toFixed(1);
    const [x1, y1, width, height] = box.bounding;

	var e = document.getElementById("linewidthbbox");
	var lwvalue = e.value;


    // draw border box
    ctx.strokeStyle = color;
    ctx.lineWidth =  parseFloat(lwvalue); //1.0;//Math.max(
    //  Math.min(ctx.canvas.width, ctx.canvas.height) / 200,
    //  2.5
    //);
    ctx.strokeRect(x1, y1, width, height);

    //// draw the label background.
    //ctx.fillStyle = color;
    //const textWidth = ctx.measureText(className + " - " + score + "%").width;
    //const textHeight = parseInt(font, 10); // base 10
    //const yText = y1 - (textHeight + ctx.lineWidth);
    //ctx.fillRect(
    //  x1 - 1,
    //  yText < 0 ? 0 : yText,
    //  textWidth + ctx.lineWidth,
    //  textHeight + ctx.lineWidth
    //);
	//
    //// Draw labels
    //ctx.fillStyle = "#ffffff";
    //ctx.fillText(
    //  className + " - " + score + "%",
    //  x1 - 1,
    //  yText < 0 ? 1 : yText + 1
    //);
  });
};

class Colors {
  // ultralytics color palette https://ultralytics.com/
  constructor() {
    this.palette = [
      "#FF3838",
       "#7CFC00", //"#FF9D97",
      "#FF701F",
      "#FFB21D",
      "#CFD231",
      "#48F90A",
      "#92CC17",
      "#3DDB86",
      "#1A9334",
      "#00D4BB",
      "#2C99A8",
      "#00C2FF",
      "#344593",
      "#6473FF",
      "#0018EC",
      "#8438FF",
      "#520085",
      "#CB38FF",
      "#FF95C8",
      "#FF37C7",
    ];
    this.n = this.palette.length;
  }

  get = (i) => this.palette[Math.floor(i) % this.n];

  static hexToRgba = (hex, alpha) => {
    let result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? [
          parseInt(result[1], 16),
          parseInt(result[2], 16),
          parseInt(result[3], 16),
          alpha,
        ]
      : null;
  };
}
const colors = new Colors();

// Load image
document.getElementById("imgLoader").onchange = function (evt) {
  let tgt = evt.target,
    files = tgt.files;
  if (FileReader && files && files.length) {
    let fr = new FileReader();
    fr.onload = function () {
      document.getElementById("loadedImg").src = fr.result;
      // Clear canvas before new detection
      const context = canvas.getContext("2d");
      context.clearRect(0, 0, canvas.width, canvas.height);
    };
    fr.readAsDataURL(files[0]);
  }
};

function runInference() {
  detectImage(
    document.querySelector("#loadedImg"),
    document.querySelector("canvas"),
    mySession,
    topk,
    iouThreshold,
    confThreshold,
    classThreshold,
    modelInputShape
  );
}
