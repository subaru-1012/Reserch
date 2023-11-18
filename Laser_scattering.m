%% 定数を指定
clear

% 解析領域（単位：ピクセル）
% 整数でないと警告が出る
rad= 800;

% 画素数=1322のとき、実際の長さは40mm
pixel_length = 1322;
actual_length = 40;

% １ピクセルに対応する実際の長さ
length_per_pixel=actual_length/pixel_length;

% ダーク領域の平均輝度（計測値16bit変換済み）
ave_dark=[771.88, 771.944, 771.944, 772.0048, 772.3088, 772.7248, 773.6992, 776.8656, 787.4256];

% 露光時間1000ms→1sにし、逆数に変換
exp_coeff=[10^3.5, 1000, 10^2.5, 10^2, 10^1.5, 10, 10^0.5, 1, 10^(-0.5)];

% プロファイルのノイズ除去に使用する閾値（プロファイルがうまく描画できれば変更の必要性なし）
CUT_MIN = 900;
CUT_MAX = 65000; % 65534にすると532nmの飽和領域もちゃんと出してくれる（ただし飽和領域はノイズ）

%% 測定日_MN製造日を設定する（ファイル名）
days = "230619_MN220907";

%% 解析スクリプト
tic
test = [];
sample_name = [];
for wavelength = ["red_633", "nir_850", "green_532", "nir_808", "violet_405"]% 使う波長を選択
% for wavelength = ["nir_808"]% 使う波長を選択
% for wavelength = ["red_633", "nir_850", "green_532", "nir_808"]% 使う波長を選択
    for sample_num = "01"% 個体番号を格納
        for point_num = ["1","2","3","4","5","6"]% 個体内の照射点の数 
        % for point_num = ["4"]% 個体内の照射点の数 

            %%%%%%%%%%%%%%%%%%%% loop内のpathを指定 %%%%%%%%%%%%%%%%%%%%
            % サンプル名（画像のファイル名）の定義
            % サンプル名は"個体番号_照射点番号"とする
            sample_name_temp = append(sample_num,'_',point_num);
    
            % 画像が保存されているディレクトリを指定
            path_folder = append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/",days,"/",wavelength);
            path_temp = append(path_folder,"/",sample_name_temp);
    
            % path_tempをカレントディレクトリへ変更
            cd(path_temp);
            
            %%%%%%%%%%%%%%%%%%%% 画像の読み込みおよびノイズ除去 %%%%%%%%%%%%%%%%%%%%
            % フォルダ内にある，以下の形式のデータを読み込む
            dirOutput_tif = dir('*.TIFF');
            dirOutput_tiff = dir('*.TIF');
            dirOutput_png = dir('*.png');
            
            Filenames_tiff = {dirOutput_tiff.name}'; 
            Filenames_tif = {dirOutput_tif.name}';
            Filenames_png = {dirOutput_png.name}';
            Filenames=cat(1,Filenames_tif,Filenames_tiff,Filenames_png);
            imagenum=size(Filenames,1);
            
            % 画像サイズを取得
            I=imread(Filenames{1,1});
            [imagesizeY,imagesizeX]=size(I);
            imagedata=zeros(imagesizeY,imagesizeX,imagenum);
            
            % フォルダ内の画像の読み込み・ノイズ除去
            disp('データを読み込み中･･･'); 
            for i=1:imagenum
                disp(Filenames{i,1})
                % 12bitの後方散乱画像を16bitに変換
                I=16*imread(Filenames{i,1});
                
                % ノイズ除去
                I_median = medfilt2(I,[3,3]); % 3×3の2次元median filter適用（ノイズ除去の目的）
                imagedata(:,:,i)=double(I_median);
              
                %ノイズ除去しない場合
                %imagedata(:,:,i)=double(I); 

            end

            % I,I_medianを保存
            % path_medianfilter = append("/Users/subaru/Desktop/");
            % imwrite(I, append(path_medianfilter,"nofilter.png"));
            % imwrite(I_median, append(path_medianfilter,"medianfilter.png"));

            %%%%%%%%%%%%%%%%%%%% 中心点の検出（新しい方法）%%%%%%%%%%%%%%%%%%%%
            % まずは2値化(大津法)
            temp = cast(imagedata(:,:,3),"uint16"); % データ型の変更
            temp_bina = imbinarize(temp); % 2値化
            temp_bina = imfill(temp_bina,'holes'); % 穴の埋め込み
     
            % オブジェクト検出
            stats = regionprops(temp_bina,'area','centroid'); % オブジェクトの検出結果を格納
            circ_ind=[stats.Area] >= max([stats.Area]); % 検出されたオブジェクトの中で最大面積（入射点付近)のものを算出
     
            % 入射点の位置（中心点）を算出
            circ=stats(circ_ind,:);
            center = floor(circ.Centroid);
     
            %%%%%%%%%%%%%%%%%%%% 中心点との各画素の距離・輝度（画素値）を算出　%%%%%%%%%%%%%%%%%%%%
            % 解析領域(ROI)に基づいた画像の整形
            analysis_image=squeeze(imagedata(center(2)-rad:center(2)+rad,...
                center(1)-rad:center(1)+rad,:));
     
            % 中心点からの距離における輝度値を算出
            dist_int_matrix=zeros((rad*2+1)^2,imagenum+1);
            for i=1:size(analysis_image,1)
                for j=1:size(analysis_image,2)
                    dist_int_matrix((i-1)*(rad*2+1)+j,1)=((i-rad)^2+(j-rad)^2)^(1/2);
                    for k=1:imagenum
                        % MAXの値より大きい値はNaNに置換（ノイズ除去に有効）
                        if analysis_image(i,j,k)>CUT_MAX
                            dist_int_matrix((i-1)*(rad*2+1)+j,k+1)=nan;
                        % MINの値より小さい値はNaNに置換（ノイズ除去に有効）
                        elseif analysis_image(i,j,k)<CUT_MIN
                            dist_int_matrix((i-1)*(rad*2+1)+j,k+1)=nan;
                        else
                            dist_int_matrix((i-1)*(rad*2+1)+j,k+1)=analysis_image(i,j,k);  
                        end
                    end
                end
            end
    
            % 中心点からの距離の順に並べる（ソート）
            dist_int_matrix_sort=sortrows(dist_int_matrix,1);
    
            % 同じ距離における輝度値を平均化
            % Tableデータに変換（平均値の計算がしやすくため）
            varnames={'Distance','Intensity'};
            dist_int_matrix_T=table(dist_int_matrix_sort(:,1),dist_int_matrix_sort(:,2:end),'VariableNames',varnames);
            func=@(x) mean(x,1);
            dist_meanint=varfun(func,dist_int_matrix_T,'InputVariables','Intensity',...
                'GroupingVariables','Distance');
            distance=dist_meanint.Distance;
            actual_distance = distance * length_per_pixel;
     
            %%%%%%%%%%%%%%%%%%%% 露光時間での規格化およびプロファイルのHDR合成　%%%%%%%%%%%%%%%%%%%%
            % ここではまだ露光時間が異なる輝度プロファイルは別のデータとして保存されている
            % プロファイル作成のために、各露光時間での暗電流ノイズを各画像から差し引く（暗電流は露光時間に比例しないため）
    
            % dark除去前のプロファイルを保存する
            writematrix(dist_meanint.Fun_Intensity, append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/intprofile/",days,"/",wavelength,"/intprofile0.csv"));    
    
            % ハイダイナミックレンジ画像（HDR画像)作成のために、各露光時間での暗電流ノイズを各画像から差し引く
            intprofile=dist_meanint.Fun_Intensity-repmat(ave_dark,[size(distance,1) 1]);
            intprofile(intprofile(:)<0)=nan;
     
            % dark除去後のプロファイルを保存する
            writematrix(intprofile, append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/intprofile/",days,"/",wavelength,"/intprofile1.csv"));
    
            % dark除去後×各露光時間の逆数のプロファイルを保存する
            writematrix(intprofile .* repmat(exp_coeff,[size(distance,1) 1]), append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/intprofile/",days,"/",wavelength,"/intprofile2.csv"));
     
            % log10(dark除去後×各露光時間の逆数)のプロファイルを保存する
            writematrix(log10(intprofile .* repmat(exp_coeff,[size(distance,1) 1])), append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/intprofile/",days,"/",wavelength,"/intprofile3.csv"));
    
            % プロファイルの作成 (各露光時間の逆数×各画像の平均）
            % allintprofile_log: log変換後
            allintprofile_log=mean(log10(intprofile .* repmat(exp_coeff,[size(distance,1) 1])),2,'omitnan');
     
            % 解析結果を格納
            test = [test,allintprofile_log];
            sample_name = [sample_name sample_name_temp];
    
            %%%%%%%%%%%%%%%%%%%% HDR画像の保存（MATLABの関数）　%%%%%%%%%%%%%%%%%%%%）
            imagedata_hdr = zeros(imagesizeY,imagesizeX,imagenum);
            % 各画像から対応する暗電流を除去し、uint16に整形
            for i=1:imagenum
                imagedata_hdr(:,:,i) = (imagedata(:,:,i) - ave_dark(:,i));
                imagedata_hdr(:,:,i) = uint16(imagedata_hdr(:,:,i));
            end
     
            % 解析のためにimagedata_hdrをcell形式に変更
            imagedata_hdr_cell = {};
            for i=1:imagenum
                imagedata_hdr_cell{i} = imagedata_hdr(:,:,i);
                imagedata_hdr_cell{i} = uint16(imagedata_hdr(:,:,i));
            end
    
            % HDR合成
            image_HDR = makehdr(imagedata_hdr_cell,'RelativeExposure',1./exp_coeff);
    
            % トーンマッピング：HDR画像をPCでの表示に適した形式（LDR）に変換する
            % tonemap:HDR画像をuint8クラスのRGBイメージに変換
            image_HDR1 = tonemap(image_HDR);
     
            % tonemapfarbman:既定の引数値で使用して変換を繰り返す。tonemapによるLDRよりも色の彩度が高くなる。
            image_HDR2 = tonemapfarbman(image_HDR);
     
            % localtonemap:局所的なコントラストを保持
            image_HDR3 = localtonemap(image_HDR);
     
            % 露光融合
            image_blend = blendexposure(uint16(imagedata_hdr(:,:,1)), uint16(imagedata_hdr(:,:,2)), uint16(imagedata_hdr(:,:,3)), uint16(imagedata_hdr(:,:,4)), uint16(imagedata_hdr(:,:,5)), uint16(imagedata_hdr(:,:,6)), uint16(imagedata_hdr(:,:,7)), uint16(imagedata_hdr(:,:,8)), uint16(imagedata_hdr(:,:,9)));
     
            % 解析に使用する範囲はおおよそ、動径距離が35mmまでのため、それ用に加工する
            %image_HDR = image_HDR(:,512:3584);
    
            % HDR画像の保存
            % HDR(tonemap)
            path_hdr1 = append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/HDR1/",days,"/",wavelength,"/");
            imwrite(image_HDR1, append(path_hdr1,sample_name_temp,".png"));
     
            % HDR(tonemapfarbman)
            path_hdr2 = append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/HDR2/",days,"/",wavelength,"/");
            imwrite(image_HDR2, append(path_hdr2,sample_name_temp,".png"));
    
            % HDR(localtonemap)
            path_hdr3 = append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/HDR3/",days,"/",wavelength,"/");
            imwrite(image_HDR3, append(path_hdr3,sample_name_temp,".png"));
     
            % 露光融合画像の保存
            path_blend = append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/Blend/",days,"/",wavelength,"/");
            imwrite(image_blend, append(path_blend,sample_name_temp,".png"));
            clc
        end 
    end
     
    % csv化のための成型
    test = [sample_name;test];
    actual_distance = [("distance (mm)"); actual_distance];
    test = [actual_distance test];
     
    % プロファイルの保存
    writematrix(test,append("/Users/subaru/Desktop/Reserch/Laserscattering/experiment/Profile/",days,"/","Profile_",wavelength,".csv"))

    % 変数を初期化
    test = [];
    sample_name = [];
    path_temp = [];

end

toc
disp("finish")
