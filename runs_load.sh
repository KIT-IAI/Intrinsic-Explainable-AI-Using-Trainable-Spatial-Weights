#!/usr/bin/env sh

# early fusion raw
# python src/large_scale_forecasting.py targets=[de_transnetbw+load] name="transnetbw_load" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[de_50hertz+load] name="50hertz_load" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[de+load] name="de_load" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[gb+load] name="gb_load" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[es+load] name="es_load" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[fr+load] name="fr_load" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+load] name="eu_load" sweep="efm_raw" data/weather=era5 model=early_fusion_model verbose=True

# early fusion statistics
# python src/large_scale_forecasting.py targets=[de_transnetbw+load] name="transnetbw_load" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[de_50hertz+load] name="50hertz_load" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[de+load] name="de_load" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[gb+load] name="gb_load" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[es+load] name="es_load" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[fr+load] name="fr_load" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[eu+load] name="eu_load" sweep="efm_statistics" data/weather=era5_statistics model=early_fusion_model verbose=True

# early fusion pca
python src/large_scale_forecasting.py targets=[de_transnetbw+load] name="transnetbw_load" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+load] name="50hertz_load" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+load] name="de_load" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+load] name="gb_load" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+load] name="es_load" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+load] name="fr_load" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[eu+load] name="eu_load" sweep="efm_pca" data/weather=era5_pca model=early_fusion_model verbose=True

# late fusion raw
python src/large_scale_forecasting.py targets=[de_transnetbw+load] name="transnetbw_load" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+load] name="50hertz_load" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+load] name="de_load" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+load] name="gb_load" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+load] name="es_load" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+load] name="fr_load" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+load] name="eu_load" sweep="lfm_raw" data/weather=era5 model=late_fusion_model verbose=True

# multi modal fusion raw
python src/large_scale_forecasting.py targets=[de_transnetbw+load] name="transnetbw_load" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+load] name="50hertz_load" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+load] name="de_load" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+load] name="gb_load" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+load] name="es_load" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+load] name="fr_load" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+load] name="eu_load" sweep="mmfm_raw" data/weather=era5 model=multi_modal_fusion_model verbose=True

# late fusion statistics
python src/large_scale_forecasting.py targets=[de_transnetbw+load] name="transnetbw_load" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+load] name="50hertz_load" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+load] name="de_load" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+load] name="gb_load" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+load] name="es_load" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+load] name="fr_load" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+load] name="eu_load" sweep="lfm_statistics" data/weather=era5_statistics model=late_fusion_model verbose=True

# multi modal fusion statistics
python src/large_scale_forecasting.py targets=[de_transnetbw+load] name="transnetbw_load" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+load] name="50hertz_load" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+load] name="de_load" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+load] name="gb_load" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+load] name="es_load" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+load] name="fr_load" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+load] name="eu_load" sweep="mmfm_statistics" data/weather=era5_statistics model=multi_modal_fusion_model verbose=True

# late fusion pca
python src/large_scale_forecasting.py targets=[de_transnetbw+load] name="transnetbw_load" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+load] name="50hertz_load" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+load] name="de_load" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+load] name="gb_load" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+load] name="es_load" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+load] name="fr_load" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[eu+load] name="eu_load" sweep="lfm_pca" data/weather=era5_pca model=late_fusion_model verbose=True

# multi modal fusion pca
python src/large_scale_forecasting.py targets=[de_transnetbw+load] name="transnetbw_load" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+load] name="50hertz_load" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+load] name="de_load" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+load] name="gb_load" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+load] name="es_load" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+load] name="fr_load" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True
# python src/large_scale_forecasting.py targets=[eu+load] name="eu_load" sweep="mmfm_pca" data/weather=era5_pca model=multi_modal_fusion_model verbose=True

# alpha combined fusion raw
python src/large_scale_forecasting.py targets=[de_transnetbw+load] name="transnetbw_load" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+load] name="50hertz_load" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+load] name="de_load" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+load] name="gb_load" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+load] name="es_load" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+load] name="fr_load" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+load] name="eu_load" sweep="acfm_raw" data/weather=era5 model=alpha_combined_fusion_model verbose=True

# alpha separated fusion raw
python src/large_scale_forecasting.py targets=[de_transnetbw+load] name="transnetbw_load" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de_50hertz+load] name="50hertz_load" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[de+load] name="de_load" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[gb+load] name="gb_load" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[es+load] name="es_load" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[fr+load] name="fr_load" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True
python src/large_scale_forecasting.py targets=[eu+load] name="eu_load" sweep="asfm_raw" data/weather=era5 model=alpha_separated_fusion_model verbose=True