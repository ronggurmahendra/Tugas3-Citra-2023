filename = app.FileInputnameEditField.Value
% Buka file
inputImage = imread(filename);
% konversi ke grayscale apabila rgb
if ndims(inputImage) == 2
    inputImageGrey = inputImage
elseif ndims(inputImage) == 3
    inputImageGrey = rgb2gray(inputImage)
end
imshow(inputImageGrey, 'Parent', app.UIAxes);
% lakukan edge detection
resultImageLaplace = edgeLaplace(inputImageGrey);
resultImageLog = edgeLog(inputImageGrey);
resultImageSobel = edgeSobel(inputImageGrey);
resultImagePrewit = edgePrewit(inputImageGrey);
resultImageRoberts = edgeRoberts(inputImageGrey);
resultImageCanny = edgeCanny(inputImageGrey);

% cast ke uint8
edgeImage = uint8(resultImageCanny)
imshow(edgeImage, 'Parent', app.UIAxes_2);
labeledImage = bwlabel(edgeImage);

% tutup disconnected edges 
closedImage = imclose(labeledImage,strel('line',10,0));

% Fill edges
filledImage = imfill(closedImage, 'holes');

% filter apabila ukuran kecil
openedImage = imopen(filledImage, strel(ones(3,3)));
maskImage = bwareaopen(openedImage,1500);

imshow(maskImage, 'Parent', app.UIAxes_3);
segmentedImage = inputImage
% cut image dalam segment
if ndims(inputImage) == 2
    segmentedImage(~maskImage) = 0;
elseif ndims(inputImage) == 3
    colorMask = cat(3, maskImage, maskImage, maskImage);
    segmentedImage(~colorMask) = 0;
end

imshow(segmentedImage, 'Parent', app.UIAxes_4);

% Laplace Edge detenction (sesuai slide)
function result = edgeLaplace(img_input)
    % Menggunakan filter Laplace 
    % h = fspecial('log', [5, 5], 2);
    H = [1 1 1; 1 -8 1; 1 1 1];
    % H = [0 1 0; 1 -4 1; 0 1 0];
    result = uint8(convn(double(img_input), double(H)));

end
% Log Laplace of Gaussian
function result = edgeLog(img_input)
        % Menggunakan filter LoG (Laplace of Gaussian) dengan kernel default
        % H = [0 1 0; 1 -4 1; 0 1 0];
        [M,N]=size(img);
        h = fspecial('log');
        result = uint8(convn(double(img_input), double(h)));
        for i = 1:M
            result(i,1)=0;
        end
end

% sobel Edge detenction (sesuai slide)
function result = edgeSobel(img_input)
     % Menggunakan operator Sobel untuk mendeteksi tepi
    % sobelEdge = edge(grayImage, 'sobel');
    [M,N]=size(img);
    Sx = [-1 0 1; -2 0 2; -1 0 1];
    Sy = [1 2 1; 0 0 0; -1 -2 -1];
    Jx = convn(double(img_input), double(Sx), 'same');
    Jy = convn(double(img_input), double(Sy), 'same');
    result = uint8(sqrt(Jx.^2 + Jy.^2));
    for i = 1:M
        result(i,1)=0;
        result(i,N)=0;
    end
    for j = 1:N
        result(1,j)=0;
        result(M,j)=0;
    end
end

% Prewitt Edge Detection
function result = edgePrewit(img_input)
    % Menggunakan operator Prewitt untuk mendeteksi tepi
    % prewittEdge = edge(grayImage, 'prewitt');
    [M,N]=size(img_input);
    Px = [-1 0 1; -1 0 1; -1 0 1];
    Py = [-1 -1 -1; 0 0 0; 1 1 1];
    Jx = convn(double(img_input), double(Px), 'same');
    Jy = convn(double(img_input), double(Py), 'same');
    result = uint8(sqrt(Jx.^2 + Jy.^2));
    for i = 1:M
        result(i,1)=0;
        result(i,N)=0;
    end
    for j = 1:N
        result(1,j)=0;
        result(M,j)=0;
    end
end

% Roberts edge detenction
function result = edgeRoberts(img_input)
    % Menggunakan operator Roberts untuk mendeteksi tepi
    % robertsEdge = edge(grayImage, 'roberts'); 
    [M,N]=size(img_input);
    Rx = [1 0; 0 -1];
    Ry = [0 1; -1 0];
    Jx = convn(double(img_input), double(Rx), 'same');
    Jy = convn(double(img_input), double(Ry), 'same');
    result = uint8(sqrt(Jx.^2 + Jy.^2));
    for i = 1:M
        result(i,N)=0;
        result(i,1)=0;
    end
    for j = 1:N
        result(M,j)=0;
        result(1,j)=0;
    end
end

% using build in canny edge detection then cast it into uint
function result = edgeCanny(img_input)
     % Menggunakan deteksi tepi Canny bawaan MATLAB
    result = uint8(edge(img_input, 'canny'));
end
