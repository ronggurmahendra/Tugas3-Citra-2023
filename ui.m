filename = app.FileInputnameEditField.Value
inputImage = imread(filename);
if ndims(inputImage) == 2
    inputImageGrey = inputImage
elseif ndims(inputImage) == 3
    inputImageGrey = rgb2gray(inputImage)
end
imshow(inputImageGrey, 'Parent', app.UIAxes);
            
resultImageLaplace = edgeLaplace(inputImageGrey);
resultImageLog = edgeLog(inputImageGrey);
resultImageSobel = edgeSobel(inputImageGrey);
resultImagePrewit = edgePrewit(inputImageGrey);
resultImageRoberts = edgeRoberts(inputImageGrey);
resultImageCanny = edgeCanny(inputImageGrey);

edgeImage = uint8(resultImageCanny)
imshow(edgeImage, 'Parent', app.UIAxes_2);
labeledImage = bwlabel(edgeImage);

% Close disconnected edges 
closedImage = imclose(labeledImage,strel('line',10,0));

% Fill inside the edges
filledImage = imfill(closedImage, 'holes');

openedImage = imopen(filledImage, strel(ones(3,3)));
maskImage = bwareaopen(openedImage,1500);

imshow(maskImage, 'Parent', app.UIAxes_3);
segmentedImage = inputImage
if ndims(inputImage) == 2
    segmentedImage(~maskImage) = 0;
elseif ndims(inputImage) == 3
    colorMask = cat(3, maskImage, maskImage, maskImage);
    segmentedImage(~colorMask) = 0;
end

imshow(segmentedImage, 'Parent', app.UIAxes_4);

% Laplace Edge detenction (sesuai slide)
function result = edgeLaplace(img_input)
    % h = fspecial('log', [5, 5], 2);
    H = [1 1 1; 1 -8 1; 1 1 1];
    % H = [0 1 0; 1 -4 1; 0 1 0];
    result = uint8(convn(double(img_input), double(H)));

end
% Log Laplace of Gaussian
function result = edgeLog(img_input)
        % H = [0 1 0; 1 -4 1; 0 1 0];
        h = fspecial('log');
        result = uint8(convn(double(img_input), double(h)));
end

% sobel Edge detenction (sesuai slide)
function result = edgeSobel(img_input)
    % sobelEdge = edge(grayImage, 'sobel');
    Sx = [-1 0 1; -2 0 2; -1 0 1];
    Sy = [1 2 1; 0 0 0; -1 -2 -1];
    Jx = convn(double(img_input), double(Sx), 'same');
    Jy = convn(double(img_input), double(Sy), 'same');
    result = uint8(sqrt(Jx.^2 + Jy.^2));
end

% Prewit
function result = edgePrewit(img_input)
    % prewittEdge = edge(grayImage, 'prewitt');
    Px = [-1 0 1; -1 0 1; -1 0 1];
    Py = [-1 -1 -1; 0 0 0; 1 1 1];
    Jx = convn(double(img_input), double(Px), 'same');
    Jy = convn(double(img_input), double(Py), 'same');
    result = uint8(sqrt(Jx.^2 + Jy.^2));
end

% Roberts
function result = edgeRoberts(img_input)
    % robertsEdge = edge(grayImage, 'roberts'); 
    Rx = [1 0; 0 -1];
    Ry = [0 1; -1 0];
    Jx = convn(double(img_input), double(Rx), 'same');
    Jy = convn(double(img_input), double(Ry), 'same');
    result = uint8(sqrt(Jx.^2 + Jy.^2));
end

% Canny
function result = edgeCanny(img_input)
    result = uint8(edge(img_input, 'canny'));
end
