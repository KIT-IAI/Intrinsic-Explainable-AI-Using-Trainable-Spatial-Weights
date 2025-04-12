#!/usr/bin/env sh

# early fusion raw
python src/large_scale_forecasting.py targets=[de_transnetbw+solar] name="transnetbw_solar" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+solar] name="50hertz_solar" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+solar] name="de_solar" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+solar] name="gb_solar" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+solar] name="es_solar" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+solar] name="fr_solar" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+solar] name="eu_solar" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True

# early fusion statistics
python src/large_scale_forecasting.py targets=[de_transnetbw+solar] name="transnetbw_solar" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+solar] name="50hertz_solar" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+solar] name="de_solar" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+solar] name="gb_solar" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+solar] name="es_solar" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+solar] name="fr_solar" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+solar] name="eu_solar" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True

# early fusion pca
python src/large_scale_forecasting.py targets=[de_transnetbw+solar] name="transnetbw_solar" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+solar] name="50hertz_solar" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+solar] name="de_solar" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+solar] name="gb_solar" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+solar] name="es_solar" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+solar] name="fr_solar" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[eu+solar] name="eu_solar" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True

# late fusion raw
python src/large_scale_forecasting.py targets=[de_transnetbw+solar] name="transnetbw_solar" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+solar] name="50hertz_solar" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+solar] name="de_solar" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+solar] name="gb_solar" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+solar] name="es_solar" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+solar] name="fr_solar" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+solar] name="eu_solar" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True

# multi modal fusion raw
python src/large_scale_forecasting.py targets=[de_transnetbw+solar] name="transnetbw_solar" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+solar] name="50hertz_solar" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+solar] name="de_solar" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+solar] name="gb_solar" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+solar] name="es_solar" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+solar] name="fr_solar" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+solar] name="eu_solar" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True

# late fusion statistics
python src/large_scale_forecasting.py targets=[de_transnetbw+solar] name="transnetbw_solar" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+solar] name="50hertz_solar" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+solar] name="de_solar" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+solar] name="gb_solar" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+solar] name="es_solar" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+solar] name="fr_solar" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+solar] name="eu_solar" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True

# multi modal fusion statistics
python src/large_scale_forecasting.py targets=[de_transnetbw+solar] name="transnetbw_solar" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+solar] name="50hertz_solar" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+solar] name="de_solar" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+solar] name="gb_solar" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+solar] name="es_solar" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+solar] name="fr_solar" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+solar] name="eu_solar" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True

# late fusion pca
python src/large_scale_forecasting.py targets=[de_transnetbw+solar] name="transnetbw_solar" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+solar] name="50hertz_solar" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+solar] name="de_solar" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+solar] name="gb_solar" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+solar] name="es_solar" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+solar] name="fr_solar" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[eu+solar] name="eu_solar" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True

# multi modal fusion pca
python src/large_scale_forecasting.py targets=[de_transnetbw+solar] name="transnetbw_solar" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+solar] name="50hertz_solar" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+solar] name="de_solar" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+solar] name="gb_solar" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+solar] name="es_solar" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+solar] name="fr_solar" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[eu+solar] name="eu_solar" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True

# alpha combined fusion raw
python src/large_scale_forecasting.py targets=[de_transnetbw+solar] name="transnetbw_solar" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+solar] name="50hertz_solar" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+solar] name="de_solar" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+solar] name="gb_solar" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+solar] name="es_solar" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+solar] name="fr_solar" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+solar] name="eu_solar" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True

# alpha separated fusion raw
python src/large_scale_forecasting.py targets=[de_transnetbw+solar] name="transnetbw_solar" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+solar] name="50hertz_solar" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+solar] name="de_solar" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+solar] name="gb_solar" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+solar] name="es_solar" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+solar] name="fr_solar" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+solar] name="eu_solar" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True