"""
===============================================================================
Anomaly Detection Project: Application dedicated to Novelty Detection for
bicycle traffic metering systems in Nantes
===============================================================================
"""
# Standard library
import platform

# Other libraries
import numpy as np
import pandas as pd
import joblib
import gradio as gr


from joblib import load


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Joblib: {}'.format(joblib.__version__))
print('Gradio: {}'.format(gr.__version__))



def get_prediction(date: str, meter_id: str, meter_name: str,
                   meter_reading: int) -> str:
    """This function predicts if there are changes of events in bicycle
    traffic metering data using a trained Machine Learning (ML) model.

    Args:
        date (str): the date of metering reading
        meter_id (str): the meter identifier
        meter_name (str): the meter name
        meter_reading (int): the counting value displayed by the meter

    Returns:
        response (str): the prediction of the model
    """

    try:

        # Check wether the user inputs are valid
        if (date and meter_id and meter_name and date != meter_id and
            date != meter_name and meter_id != meter_name and
            meter_reading is not None):

            # Cleanse the text
            date = date.strip()

            # Encode the Meter ID and Meter name features for novelty detection
            encoder_path = 'models/encoder/encoder.joblib'
            encoder = load(encoder_path)
            categ_values = np.array([[meter_id, meter_name]])
            categ_values_enc = encoder.transform(categ_values)

            # Encode the Meter name feature for regression
            reg_encoder_path = 'models/encoder/reg_encoder.joblib'
            reg_encoder = load(reg_encoder_path)
            meter_name_enc = reg_encoder.transform([[meter_name]])[0][0]

            # Normalisation of Meter reading feature for regression
            scaler_path = 'models/scaler/scaler.joblib'
            scaler = load(scaler_path)
            scaled_meter_reading = scaler.transform([[meter_reading]])[0][0]

            # Prediction of Modelled value feature for novelty detection
            reg_model_path = 'models/regression/perpetualbooster/model.joblib'
            reg_model = load(reg_model_path)
            X_reg = np.array([[meter_name_enc, scaled_meter_reading]])
            modelled_value = int(reg_model.predict(X_reg)[0])

            # Create dataset with input data
            meter_id_flag = True if meter_id in [949, 950] else False
            X = pd.DataFrame(
                data={
                    'Meter ID': [categ_values_enc[:, 0]],
                    'Meter name': [categ_values_enc[:, 1]],
                    'Meter reading': [meter_reading],
                    'Modelled value': [modelled_value],
                    'Meter ID flag': [meter_id_flag]
                }
            )

            # Load the trained ML model
            model_path = 'models/novelty detection/model.joblib'
            model = load(model_path)

            # Make prediction
            prediction = model.predict(np.array(X))

            # Display the result
            if prediction[0] == 1:
                response = (f'The metering data taken on {date} from meter '
                            f'{meter_name} with ID number {meter_id} and '
                            f'the meter reading of {meter_reading} contains '
                            f'changes of events.')
            else:
                response = (f'The metering data taken on {date} from meter '
                            f'{meter_name} with ID number {meter_id} and '
                            f'the meter reading of {meter_reading} does not '
                            f'show any changes of events.')
        else:
            response = ('Invalid input data. Please complete the fields '
                        'correctly.')

    except Exception as error:
        response = f'The following unexpected error occurred: {error}'
    return response



# Instantiate the app
meter_id_list = [
    '664', '665', '668', '669', '674', '675', '682', '683', '701', '725',
    '744', '745', '785', '881', '889', '907', '947', '948', '949', '950'
]
meter_names_list = [
    'Pont A. Briand vers Sud', 'Pont A. Briand vers Nord',
    'De_Gaulle_vers_Sud', 'De_Gaulle_vers_Nord', 'Pont_Haudaudine_vers_Sud',
    'Pont_Haudaudine_vers_Nord', 'Pont_A_de_Bretagne_Nord_vers_Sud',
    'Pont_A_de_Bretagne_Sud_vers_Nord', 'Prairie_de_Mauves',
    'Pont Tabarly vers Sud', 'Calvaire_vers_Est', 'Calvaire_vers_Ouest',
    'Cours_Des_50_Otages_Sud', 'Chaussée de la Madeleine', 'Bouaye_cote_stade',
    'Bouaye_cote_maison', 'Bd Malakoff vers Ouest', 'Bd Malakoff vers Est',
    'La Chapelle sur Erdre', 'Saint Léger les Vignes'
]
app = gr.Interface(
    fn=get_prediction,
    inputs=[
        gr.Textbox(label='Date'),
        gr.Dropdown(
            choices=meter_id_list,
            label='Meter ID',
            type='value'
        ),
        gr.Dropdown(
            choices=meter_names_list,
            label='Meter name',
            type='value'
        ),
        gr.Number(
            label='Meter reading',
            minimum=0
        )
    ],
    outputs=gr.Textbox(label='Detection result'),
    title='Novelty Detection Application for Bicycle Counting System in Nantes'
)



if __name__ == '__main__':
    app.launch()
