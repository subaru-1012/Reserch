%% �萔���w��
clear

% ��͗̈�i�P�ʁF�s�N�Z���j
% �����łȂ��ƌx�����o��
rad= 800;

% ��f��=1322�̂Ƃ��A���ۂ̒�����40mm
pixel_length = 1322;
actual_length = 40;

% �P�s�N�Z���ɑΉ�������ۂ̒���
length_per_pixel=actual_length/pixel_length;

% �_�[�N�̈�̕��ϋP�x�i�v���l16bit�ϊ��ς݁j
ave_dark=[771.88, 771.944, 771.944, 772.0048, 772.3088, 772.7248, 773.6992, 776.8656, 787.4256];

% �I������1000ms��1s�ɂ��A�t���ɕϊ�
exp_coeff=[10^3.5, 1000, 10^2.5, 10^2, 10^1.5, 10, 10^0.5, 1, 10^(-0.5)];

% �v���t�@�C���̃m�C�Y�����Ɏg�p����臒l�i�v���t�@�C�������܂��`��ł���ΕύX�̕K�v���Ȃ��j
CUT_MIN = 900;
CUT_MAX = 65000; % 65534�ɂ����532nm�̖O�a�̈�������Əo���Ă����i�������O�a�̈�̓m�C�Y�j

%% �����_MN��������ݒ肷��i�t�@�C�����j
days = "230619_MN220907";

%% ��̓X�N���v�g
tic
test = [];
sample_name = [];
for wavelength = ["red_633", "nir_850", "green_532", "nir_808", "violet_405"]% �g���g����I��
% for wavelength = ["nir_808"]% �g���g����I��
% for wavelength = ["red_633", "nir_850", "green_532", "nir_808"]% �g���g����I��
    for sample_num = "01"% �̔ԍ����i�[
        for point_num = ["1","2","3","4","5","6"]% �̓��̏Ǝ˓_�̐� 
        % for point_num = ["4"]% �̓��̏Ǝ˓_�̐� 

            %%%%%%%%%%%%%%%%%%%% loop����path���w�� %%%%%%%%%%%%%%%%%%%%
            % �T���v�����i�摜�̃t�@�C�����j�̒�`
            % �T���v������"�̔ԍ�_�Ǝ˓_�ԍ�"�Ƃ���
            sample_name_temp = append(sample_num,'_',point_num);
    
            % �摜���ۑ�����Ă���f�B���N�g�����w��
            path_folder = append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/",days,"/",wavelength);
            path_temp = append(path_folder,"/",sample_name_temp);
    
            % path_temp���J�����g�f�B���N�g���֕ύX
            cd(path_temp);
            
            %%%%%%%%%%%%%%%%%%%% �摜�̓ǂݍ��݂���уm�C�Y���� %%%%%%%%%%%%%%%%%%%%
            % �t�H���_���ɂ���C�ȉ��̌`���̃f�[�^��ǂݍ���
            dirOutput_tif = dir('*.TIFF');
            dirOutput_tiff = dir('*.TIF');
            dirOutput_png = dir('*.png');
            
            Filenames_tiff = {dirOutput_tiff.name}'; 
            Filenames_tif = {dirOutput_tif.name}';
            Filenames_png = {dirOutput_png.name}';
            Filenames=cat(1,Filenames_tif,Filenames_tiff,Filenames_png);
            imagenum=size(Filenames,1);
            
            % �摜�T�C�Y���擾
            I=imread(Filenames{1,1});
            [imagesizeY,imagesizeX]=size(I);
            imagedata=zeros(imagesizeY,imagesizeX,imagenum);
            
            % �t�H���_���̉摜�̓ǂݍ��݁E�m�C�Y����
            disp('�f�[�^��ǂݍ��ݒ����'); 
            for i=1:imagenum
                disp(Filenames{i,1})
                % 12bit�̌���U���摜��16bit�ɕϊ�
                I=16*imread(Filenames{i,1});
                
                % �m�C�Y����
                I_median = medfilt2(I,[3,3]); % 3�~3��2����median filter�K�p�i�m�C�Y�����̖ړI�j
                imagedata(:,:,i)=double(I_median);
              
                %�m�C�Y�������Ȃ��ꍇ
                %imagedata(:,:,i)=double(I); 

            end

            % I,I_median��ۑ�
            % path_medianfilter = append("/Users/subaru/Desktop/");
            % imwrite(I, append(path_medianfilter,"nofilter.png"));
            % imwrite(I_median, append(path_medianfilter,"medianfilter.png"));

            %%%%%%%%%%%%%%%%%%%% ���S�_�̌��o�i�V�������@�j%%%%%%%%%%%%%%%%%%%%
            % �܂���2�l��(��Ö@)
            temp = cast(imagedata(:,:,3),"uint16"); % �f�[�^�^�̕ύX
            temp_bina = imbinarize(temp); % 2�l��
            temp_bina = imfill(temp_bina,'holes'); % ���̖��ߍ���
     
            % �I�u�W�F�N�g���o
            stats = regionprops(temp_bina,'area','centroid'); % �I�u�W�F�N�g�̌��o���ʂ��i�[
            circ_ind=[stats.Area] >= max([stats.Area]); % ���o���ꂽ�I�u�W�F�N�g�̒��ōő�ʐρi���˓_�t��)�̂��̂��Z�o
     
            % ���˓_�̈ʒu�i���S�_�j���Z�o
            circ=stats(circ_ind,:);
            center = floor(circ.Centroid);
     
            %%%%%%%%%%%%%%%%%%%% ���S�_�Ƃ̊e��f�̋����E�P�x�i��f�l�j���Z�o�@%%%%%%%%%%%%%%%%%%%%
            % ��͗̈�(ROI)�Ɋ�Â����摜�̐��`
            analysis_image=squeeze(imagedata(center(2)-rad:center(2)+rad,...
                center(1)-rad:center(1)+rad,:));
     
            % ���S�_����̋����ɂ�����P�x�l���Z�o
            dist_int_matrix=zeros((rad*2+1)^2,imagenum+1);
            for i=1:size(analysis_image,1)
                for j=1:size(analysis_image,2)
                    dist_int_matrix((i-1)*(rad*2+1)+j,1)=((i-rad)^2+(j-rad)^2)^(1/2);
                    for k=1:imagenum
                        % MAX�̒l���傫���l��NaN�ɒu���i�m�C�Y�����ɗL���j
                        if analysis_image(i,j,k)>CUT_MAX
                            dist_int_matrix((i-1)*(rad*2+1)+j,k+1)=nan;
                        % MIN�̒l��菬�����l��NaN�ɒu���i�m�C�Y�����ɗL���j
                        elseif analysis_image(i,j,k)<CUT_MIN
                            dist_int_matrix((i-1)*(rad*2+1)+j,k+1)=nan;
                        else
                            dist_int_matrix((i-1)*(rad*2+1)+j,k+1)=analysis_image(i,j,k);  
                        end
                    end
                end
            end
    
            % ���S�_����̋����̏��ɕ��ׂ�i�\�[�g�j
            dist_int_matrix_sort=sortrows(dist_int_matrix,1);
    
            % ���������ɂ�����P�x�l�𕽋ω�
            % Table�f�[�^�ɕϊ��i���ϒl�̌v�Z�����₷�����߁j
            varnames={'Distance','Intensity'};
            dist_int_matrix_T=table(dist_int_matrix_sort(:,1),dist_int_matrix_sort(:,2:end),'VariableNames',varnames);
            func=@(x) mean(x,1);
            dist_meanint=varfun(func,dist_int_matrix_T,'InputVariables','Intensity',...
                'GroupingVariables','Distance');
            distance=dist_meanint.Distance;
            actual_distance = distance * length_per_pixel;
     
            %%%%%%%%%%%%%%%%%%%% �I�����Ԃł̋K�i������уv���t�@�C����HDR�����@%%%%%%%%%%%%%%%%%%%%
            % �����ł͂܂��I�����Ԃ��قȂ�P�x�v���t�@�C���͕ʂ̃f�[�^�Ƃ��ĕۑ�����Ă���
            % �v���t�@�C���쐬�̂��߂ɁA�e�I�����Ԃł̈Ód���m�C�Y���e�摜���獷�������i�Ód���͘I�����Ԃɔ�Ⴕ�Ȃ����߁j
    
            % dark�����O�̃v���t�@�C����ۑ�����
            writematrix(dist_meanint.Fun_Intensity, append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/intprofile/",days,"/",wavelength,"/intprofile0.csv"));    
    
            % �n�C�_�C�i�~�b�N�����W�摜�iHDR�摜)�쐬�̂��߂ɁA�e�I�����Ԃł̈Ód���m�C�Y���e�摜���獷������
            intprofile=dist_meanint.Fun_Intensity-repmat(ave_dark,[size(distance,1) 1]);
            intprofile(intprofile(:)<0)=nan;
     
            % dark������̃v���t�@�C����ۑ�����
            writematrix(intprofile, append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/intprofile/",days,"/",wavelength,"/intprofile1.csv"));
    
            % dark������~�e�I�����Ԃ̋t���̃v���t�@�C����ۑ�����
            writematrix(intprofile .* repmat(exp_coeff,[size(distance,1) 1]), append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/intprofile/",days,"/",wavelength,"/intprofile2.csv"));
     
            % log10(dark������~�e�I�����Ԃ̋t��)�̃v���t�@�C����ۑ�����
            writematrix(log10(intprofile .* repmat(exp_coeff,[size(distance,1) 1])), append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/intprofile/",days,"/",wavelength,"/intprofile3.csv"));
    
            % �v���t�@�C���̍쐬 (�e�I�����Ԃ̋t���~�e�摜�̕��ρj
            % allintprofile_log: log�ϊ���
            allintprofile_log=mean(log10(intprofile .* repmat(exp_coeff,[size(distance,1) 1])),2,'omitnan');
     
            % ��͌��ʂ��i�[
            test = [test,allintprofile_log];
            sample_name = [sample_name sample_name_temp];
    
            %%%%%%%%%%%%%%%%%%%% HDR�摜�̕ۑ��iMATLAB�̊֐��j�@%%%%%%%%%%%%%%%%%%%%�j
            imagedata_hdr = zeros(imagesizeY,imagesizeX,imagenum);
            % �e�摜����Ή�����Ód�����������Auint16�ɐ��`
            for i=1:imagenum
                imagedata_hdr(:,:,i) = (imagedata(:,:,i) - ave_dark(:,i));
                imagedata_hdr(:,:,i) = uint16(imagedata_hdr(:,:,i));
            end
     
            % ��͂̂��߂�imagedata_hdr��cell�`���ɕύX
            imagedata_hdr_cell = {};
            for i=1:imagenum
                imagedata_hdr_cell{i} = imagedata_hdr(:,:,i);
                imagedata_hdr_cell{i} = uint16(imagedata_hdr(:,:,i));
            end
    
            % HDR����
            image_HDR = makehdr(imagedata_hdr_cell,'RelativeExposure',1./exp_coeff);
    
            % �g�[���}�b�s���O�FHDR�摜��PC�ł̕\���ɓK�����`���iLDR�j�ɕϊ�����
            % tonemap:HDR�摜��uint8�N���X��RGB�C���[�W�ɕϊ�
            image_HDR1 = tonemap(image_HDR);
     
            % tonemapfarbman:����̈����l�Ŏg�p���ĕϊ����J��Ԃ��Btonemap�ɂ��LDR�����F�̍ʓx�������Ȃ�B
            image_HDR2 = tonemapfarbman(image_HDR);
     
            % localtonemap:�Ǐ��I�ȃR���g���X�g��ێ�
            image_HDR3 = localtonemap(image_HDR);
     
            % �I���Z��
            image_blend = blendexposure(uint16(imagedata_hdr(:,:,1)), uint16(imagedata_hdr(:,:,2)), uint16(imagedata_hdr(:,:,3)), uint16(imagedata_hdr(:,:,4)), uint16(imagedata_hdr(:,:,5)), uint16(imagedata_hdr(:,:,6)), uint16(imagedata_hdr(:,:,7)), uint16(imagedata_hdr(:,:,8)), uint16(imagedata_hdr(:,:,9)));
     
            % ��͂Ɏg�p����͈͂͂����悻�A���a������35mm�܂ł̂��߁A����p�ɉ��H����
            %image_HDR = image_HDR(:,512:3584);
    
            % HDR�摜�̕ۑ�
            % HDR(tonemap)
            path_hdr1 = append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/HDR1/",days,"/",wavelength,"/");
            imwrite(image_HDR1, append(path_hdr1,sample_name_temp,".png"));
     
            % HDR(tonemapfarbman)
            path_hdr2 = append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/HDR2/",days,"/",wavelength,"/");
            imwrite(image_HDR2, append(path_hdr2,sample_name_temp,".png"));
    
            % HDR(localtonemap)
            path_hdr3 = append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/HDR3/",days,"/",wavelength,"/");
            imwrite(image_HDR3, append(path_hdr3,sample_name_temp,".png"));
     
            % �I���Z���摜�̕ۑ�
            path_blend = append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/Blend/",days,"/",wavelength,"/");
            imwrite(image_blend, append(path_blend,sample_name_temp,".png"));
            clc
        end 
    end
     
    % csv���̂��߂̐��^
    test = [sample_name;test];
    actual_distance = [("distance (mm)"); actual_distance];
    test = [actual_distance test];
     
    % �v���t�@�C���̕ۑ�
    writematrix(test,append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/Profile/",days,"/","Profile_",wavelength,".csv"))

    % �ϐ���������
    test = [];
    sample_name = [];
    path_temp = [];

end

toc
disp("finish")
