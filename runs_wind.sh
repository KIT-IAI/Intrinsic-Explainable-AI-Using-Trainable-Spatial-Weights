#!/usr/bin/env sh

# early fusion raw
python src/large_scale_forecasting.py targets=[de_transnetbw+wind] name="transnetbw_wind" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+wind] name="50hertz_wind" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+wind] name="de_wind" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+wind] name="gb_wind" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+wind] name="es_wind" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+wind] name="fr_wind" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+wind] name="eu_wind" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True

# early fusion statistics
python src/large_scale_forecasting.py targets=[de_transnetbw+wind] name="transnetbw_wind" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+wind] name="50hertz_wind" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+wind] name="de_wind" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+wind] name="gb_wind" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+wind] name="es_wind" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+wind] name="fr_wind" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+wind] name="eu_wind" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True

# early fusion pca
python src/large_scale_forecasting.py targets=[de_transnetbw+wind] name="transnetbw_wind" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+wind] name="50hertz_wind" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+wind] name="de_wind" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+wind] name="gb_wind" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+wind] name="es_wind" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+wind] name="fr_wind" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[eu+wind] name="eu_wind" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True

# late fusion raw
python src/large_scale_forecasting.py targets=[de_transnetbw+wind] name="transnetbw_wind" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+wind] name="50hertz_wind" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+wind] name="de_wind" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+wind] name="gb_wind" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+wind] name="es_wind" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+wind] name="fr_wind" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+wind] name="eu_wind" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True

# multi modal fusion raw
python src/large_scale_forecasting.py targets=[de_transnetbw+wind] name="transnetbw_wind" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+wind] name="50hertz_wind" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+wind] name="de_wind" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+wind] name="gb_wind" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+wind] name="es_wind" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+wind] name="fr_wind" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+wind] name="eu_wind" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True

# late fusion statistics
python src/large_scale_forecasting.py targets=[de_transnetbw+wind] name="transnetbw_wind" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+wind] name="50hertz_wind" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+wind] name="de_wind" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+wind] name="gb_wind" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+wind] name="es_wind" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+wind] name="fr_wind" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+wind] name="eu_wind" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True

# multi modal fusion statistics
python src/large_scale_forecasting.py targets=[de_transnetbw+wind] name="transnetbw_wind" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+wind] name="50hertz_wind" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+wind] name="de_wind" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+wind] name="gb_wind" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+wind] name="es_wind" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+wind] name="fr_wind" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+wind] name="eu_wind" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True

# late fusion pca
python src/large_scale_forecasting.py targets=[de_transnetbw+wind] name="transnetbw_wind" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+wind] name="50hertz_wind" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+wind] name="de_wind" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+wind] name="gb_wind" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+wind] name="es_wind" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+wind] name="fr_wind" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[eu+wind] name="eu_wind" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True

# multi modal fusion pca
python src/large_scale_forecasting.py targets=[de_transnetbw+wind] name="transnetbw_wind" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+wind] name="50hertz_wind" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+wind] name="de_wind" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+wind] name="gb_wind" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+wind] name="es_wind" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+wind] name="fr_wind" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[eu+wind] name="eu_wind" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True

# alpha combined fusion raw
python src/large_scale_forecasting.py targets=[de_transnetbw+wind] name="transnetbw_wind" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+wind] name="50hertz_wind" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+wind] name="de_wind" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+wind] name="gb_wind" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+wind] name="es_wind" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+wind] name="fr_wind" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+wind] name="eu_wind" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True

# alpha separated fusion raw
python src/large_scale_forecasting.py targets=[de_transnetbw+wind] name="transnetbw_wind" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+wind] name="50hertz_wind" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+wind] name="de_wind" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+wind] name="gb_wind" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+wind] name="es_wind" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+wind] name="fr_wind" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+wind] name="eu_wind" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True