function [batch_G, batch_D, gt_D, mask] = GFC_Gen_training_D(parm)
% data layer
batch_G = single(zeros(parm.patchsize,parm.patchsize,3,parm.batchsize));
batch_D = single(zeros(parm.patchsize,parm.patchsize,3,parm.batchsize));
gt_D = single(ones(1, 1, 1, parm.batchsize));

idpool = randperm(parm.train_num);
count = 1;
parm.interval = 1;
mask = zeros (parm.interval^2,2,parm.batchsize);

while count <= parm.batchsize
   idx = idpool(count);
   img = imread(fullfile(parm.train_folder,parm.trainlst{idx}));
   img = (imresize(img, [parm.patchsize,parm.patchsize]));

   [r,c,cha] = size(img);
   
   if cha < 3
       img = repmat(img, [1 1 3]);
   end
  
   rate =  (rand - 0.5)/10;
   shift_x = floor(max(r,c) * rate);
   rate =  (rand - 0.5)/10;
   shift_y = floor(max(r,c) * rate);
   scale_x = 1;
   scale_y =  scale_x;
   angle = (rand - 0.5)*(10/180)*pi;
   A = [scale_x * cos(angle), scale_y * sin(angle), shift_x;...
    -scale_x * sin(angle), scale_y * cos(angle), shift_y]';
   T = maketform('affine', A);
   simg = single(imtransform(img, T, 'XData',[1,c], 'YData',[1,r], 'FillValues',127));
      
   img = simg;
   img = (img - min(img(:))) ./ (max(img(:))-min(img(:)));
   img = -1 + 2 * img;
   
   margin_x = parm.patchsize - parm.masksize -10;
   margin_y = parm.patchsize - parm.masksize -10;
   
   output = img;
   img1 = fliplr(img);
   output1 = img1;
   p = 1;
   for i=4:margin_x/parm.interval:margin_x
       for j=4:margin_y/parm.interval:margin_y
           rand_x = ceil(rand * margin_x/parm.interval+i);  
           rand_y = ceil(rand * margin_y/parm.interval+j);
                                 
           img(max(1,rand_y):max(1,rand_y)+parm.masksize-1, ...
               max(1,rand_x):max(1,rand_x)+parm.masksize-1,:) = single(-1+2*rand(parm.masksize,parm.masksize,3));
           
           img1(max(1,rand_y):max(1,rand_y)+parm.masksize-1, ...
               max(1,rand_x):max(1,rand_x)+parm.masksize-1,:) = single(-1+2*rand(parm.masksize,parm.masksize,3));
           
           mask(p,1,count) = rand_x;mask(p,1,count+1) = rand_x;
           mask(p,2,count) = rand_y;mask(p,2,count+1) = rand_y;
           p = p + 1;
       end
   end
   
   batch_G(:,:,1:3,count) = img;
   batch_G(:,:,1:3,count+1) = img1;
 
   batch_D(:,:,:,count) = output;
   batch_D(:,:,:,count+1) = output1;
   count = count + 2;
end
end
