function main()
    clc;
    close all;
    clear all;
    inputImage = imread('img3.jpg');
    if ndims(inputImage) == 2
        inputImageGrey = inputImage
    elseif ndims(inputImage) == 3
        inputImageGrey = rgb2gray(inputImage)
    end
 %   inputImageGrey = rgb2gray(inputImage)
%    inputImageGrey = inputImage

    resultImageLaplace = edgeLaplace(inputImageGrey);
    resultImageLog = edgeLog(inputImageGrey);
    resultImageSobel = edgeSobel(inputImageGrey);
    resultImagePrewit = edgePrewit(inputImageGrey);
    resultImageRoberts = edgeRoberts(inputImageGrey);
    resultImageCanny = edgeCanny(inputImageGrey);

    edgeImage = uint8(resultImageCanny)

    % closedImage = imclose(edgeImage,strel('line',10,0));

    % % Fill inside the edges
    % filledImage = imfill(closedImage, 'holes');

    % openedImage = imopen(filledImage, strel(ones(3,3)));
    % maskImage = bwareaopen(openedImage,1500);

    % resultImage = inputImage
    % red_processed = resultImage(:,:,1).*uint8(maskImage);
    % green_processed = resultImage(:,:,2).*uint8(maskImage);
    % blue_processed = resultImage(:,:,3).*uint8(maskImage);
    % segmentedImage = cat(3, red_processed, green_processed, blue_processed);

    labeledImage = bwlabel(edgeImage);

    % Close disconnected edges 
    closedImage = imclose(labeledImage,strel('line',10,0));
    
    % Fill inside the edges
    filledImage = imfill(closedImage, 'holes');

    openedImage = imopen(filledImage, strel(ones(3,3)));
    maskImage = bwareaopen(openedImage,1500);
    
    % stats = regionprops(labeledImage, 'Area', 'BoundingBox', 'Centroid');
    % areaThreshold = 0.5; 

    % for i = 1:length(stats)
    %     if stats(i).Area < areaThreshold
    %         labeledImage(labeledImage == i) = 0;
    %     end
    % end

    % labeledImage = bwlabel(labeledImage)
    % labeledImage = imfill(labeledImage, 4,"holes");
    
    
    segmentedImage = inputImage
    if ndims(inputImage) == 2
        segmentedImage(~maskImage) = 0;
    elseif ndims(inputImage) == 3
        colorMask = cat(3, maskImage, maskImage, maskImage);
        segmentedImage(~colorMask) = 0;
    end
   
    
    figure;
    imshow(inputImage)
    title('inputImage');
    figure;
    imshow(resultImageLaplace)
    title('resultImageLaplace');
    figure;
    imshow(resultImageLog)
    title('resultImageLog');
    figure;
    imshow(resultImageSobel)
    title('resultImageSobel');
    figure;
    imshow(resultImagePrewit)
    title('resultImagePrewit');
    figure;
    imshow(resultImageRoberts)
    title('resultImageRoberts');
    figure;
    imshow(resultImageCanny)
    title('resultImageCanny');
    figure;
    imshow(labeledImage);
    title('labeledImage');
    figure;
    imshow(segmentedImage);
    title('segmentedImage');
end

% Laplace
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

% sobel
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
