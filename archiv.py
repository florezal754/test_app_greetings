#flask dependencies 
from flask import Flask, jsonify, render_template, request as req
from flask_cors import CORS
import json 
import pandas as pd
#import methodology
import plotly
import plotly.graph_objects as go
import plotly.express as px
from flask_cors import CORS
from plotly.subplots import make_subplots 

#methodology dependencies
import requests 
import pandas as pd
import numpy 
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)
CORS(app)



### GRAPHS 
def my_plot_full_bar(dataframe):
    data_plot = px.bar(dataframe,x=dataframe.index,y=dataframe.columns, 
                       labels={'PBF_datetime':'Hour', 'value':'MWh', 'PBF_shortname':'Sources'},
                       title='SCHEDULED GENERATION')
    fig_json = data_plot.to_json()
    graphJSON = json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def F(dataframe):
    data_plot = px.line(dataframe,x=dataframe.index,y=dataframe.columns, 
                        title='Modulation signals',
                        labels={'PBF_datetime':'Hour','value':'Modulation signals',
                                'PBF_shortname':'Signals'})
    fig_json = data_plot.to_json()
    graphJSON = json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def pie_plot(dataframe):
    data_plot = px.pie(dataframe, values=dataframe['Total'], names="Renewables",
             color_discrete_sequence=px.colors.sequential.RdBu,
             opacity=0.7, hole=0.5)
    fig_json = data_plot.to_json()
    graphJSON = json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def pie_subplots(dataframe):
    res = dataframe['RES-E-RATIO'].mean()
    nonres = 1 - res

    wind = dataframe['Wind'].mean()
    hydro = dataframe['Hydro'].mean()
    PV = dataframe['Photovoltaic'].mean()
    thermal = dataframe['Solar thermal'].mean()
    biogas = dataframe['Biogas'].mean()
    biomass = dataframe['Biomass'].mean()

    res_df_pie = pd.DataFrame({'Type': ['Renewables', 'Non Renewables'],
                            'Percentage': [res, nonres]})

    res_pie = pd.DataFrame({'Type': ['Wind', 'Hydro', 'Photovoltaic', 'Solar thermal', 'Biogas', 'Biomass'],
                            'Percentage': [wind,hydro,PV,thermal,biogas,biomass]})

    #create subplots
    fig = make_subplots(rows=1,cols=2,specs=[[{'type':'domain'},{'type':'domain'}]])

    #creating our pie charts
    fig.add_trace(go.Pie(labels=res_df_pie['Type'], values=res_df_pie['Percentage'], name=''),1,1)
    fig.add_trace(go.Pie(labels=res_pie['Type'], values=res_pie['Percentage'], name='RES SHARE'),1,2)

    #use hole a donut like
    fig.update_traces(hole=.6, hoverinfo="label+percent+name")

    fig.update_layout(
            title_text="Mapped technologies",
            #add annotations in the center of the donnut
            annotations=[dict(text='', x=0.18, y=0.5, font_size=20, showarrow=False),
                        dict(text='', x=0.80, y=0.5, font_size=20, showarrow=False)])

    fig_json = fig.to_json()
    graphJSON = json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def dropdown_menu_line(dataframe):    
    x = dataframe.index  
    y1 = dataframe['AEF'].values.tolist()
    y2 = dataframe['APEF'].values.tolist()
    y3 = dataframe['MEFmodel'].values.tolist()
    y4 = dataframe['MPEFmodel'].values.tolist()

    
    plot = go.Figure(data=[
                        go.Scatter(name='Average Emissions Factor (AEF) [kgCO2/kWh]',x=x,y=y1),
                        go.Scatter(name='Average Primary Energy Factor (APEF) [kWpe/kWh]',x=x,y=y2),
                        go.Scatter(name='Marginal Emissions Factor (MEF) [kgCO2/kWh]',x=x,y=y3),
                        go.Scatter(name='Marginal Primary Energy Factor (MPEF) [kWpe/kWh]',x=x,y=y4)])

    # Add dropdown
    plot.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=list([
                    dict(label="Choose",
                        method="update",
                        args=[{"visible": [True, True, True, True]},
                            {"title": "Modulation signals (Double click to hide)"}]),
                    dict(label="AEF",
                        method="update",
                        args=[{"visible": [True, False, False, False]},
                            {"title": "Average Emissions Factor",
                                }]),
                    dict(label="APEF",
                        method="update",
                        args=[{"visible": [False, True, False, False]},
                            {"title": "Average Primary Energy Factor",
                                }]),
                    dict(label="MEF",
                        method="update",
                        args=[{"visible": [False, False, True, False]},
                            {"title": "Marginal Emissions Factor",
                                }]),
                    dict(label="MPEF",
                        method="update",
                        args=[{"visible": [False, False,False,True]},
                            {"title": "Marginal Primary Energy Factor",
                                }]),
                    dict(label="Emissions",
                        method="update",
                        args=[{"visible": [True, False,True,False]},
                            {"title": "Emissions (Average vs. Marginal)",
                                }]),
                    dict(label="Primary Energy",
                        method="update",
                        args=[{"visible": [False, True,False,True]},
                            {"title": "Primary Energy (Average vs. Marginal)",
                                }]),
                ]),
            )
        ])

    fig_json = plot.to_json()
    graphJSON = json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


### FUNCTIONS
def get_values(df):
    #make API call
    TOKEN = open('Token.txt', 'r').read()
    headers = {'content-Type': 'application/json', 'Authorization': 'Token token={}'.format(TOKEN)}
    #Miurl = "https://api.esios.ree.es/indicators/3"
    #print(Miurl)
    
    market_info = [3,4,9,14,15,18,21,22,10064,10073,10086,10095,10167,10258,28,29,25,26,10104,10113,10122,10186,10141]
    #31 Import andorra 
    #30,16,10196
    print("Total of indicators: ", len(market_info))
    urls = []
    data = []
    
    for ids in market_info:
        Miurl = "https://api.esios.ree.es/indicators/"+str(ids)
        urls.append(Miurl)
        #print(Miurl)
        response = requests.get(Miurl, headers=headers).json()
        indicators = response['indicator']['short_name']
        print(indicators)
        #time.sleep(1)
        #print(response)
        
        for value in response['indicator']['values']:
            PBF_value = value['value']
            PBF_datetime = value['datetime']
            #PBF_datetime_utc = value['datetime_utc']
            #PBF_tz_time = value['tz_time']
            PBF_datetime = PBF_datetime.replace('Z', "")
            PBF_shortname = response['indicator']['short_name']
            #PBF_hour = PBF_datetime.replace('2022-05-14T', "")
            #PBF_hour = PBF_hour.replace(':00:00', "")

            #saving to pandas dataframe
            df = df.append({'PBF_shortname':PBF_shortname,
                            'PBF_value':PBF_value, 
                            'PBF_datetime':PBF_datetime},ignore_index=True)
                            #'PBF_datetime_utc':PBF_datetime_utc,
                            #'PBF_tz_time':PBF_tz_time}, ignore_index=True)

        
    return df 
       
def set_datetime_index(df):
    df.PBF_datetime = pd.to_datetime(df.PBF_datetime, utc=True)
    df = df.set_index(df.PBF_datetime)
    df = df.drop('PBF_datetime', axis=1)
    print('Datetimed')
    return df 

def pivot_sort_and_fill(df):
    #short df per date
    df = pd.pivot_table(df, values='PBF_value', index=['PBF_datetime'],columns=['PBF_shortname'])
    df = df.sort_values(by=['PBF_datetime']).fillna(0)
    print('Sorted')
    return df

def rename_en(df):
    df = df.rename(columns={'Biogás':'Biogas',
                        'Biomasa':'Biomass',
                        'Carbón':'Coal',
                        'Ciclo combinado':'Combined Gas Cycle',
                        'Cogeneración':'Natural Gas Cogeneration',
                        'Consumo bombeo':'Pump Consumption',
                        'Demanda Peninsular':'Peninsular demand',
                        'Derivados del petróleo ó carbón':'Fuel',
                        'Enlace Baleares':'Balearic link',
                        'Eólica':'Wind',
                        'Generación PBF total':'Total generation',
                        'Hidráulica':'Hydro',
                        'Importación Francia':'Import France',
                        'Importación Marruecos':'Import Morocco',
                        'Importación Portugal':'Import Portugal',
                        'Nuclear':'Nuclear',
                        'Océano y geotérmica':'Ocean/Geothermal',
                        'Residuos':'Waste RSU',
                        'Saldo Marruecos':'Morocco Exchange',
                        'Saldo Portugal':'Portugal Exchange',
                        'Saldo Francia':'France Exchange',
                        'Saldo interconexiones':'Interconnections exchange',
                        'Solar fotovoltaica':'Photovoltaic',
                        'Solar térmica':'Solar thermal',
                        'Turbinación bombeo':'Pumped hydro'
                       })
    print('Renamed')
    return df 

def build_df(df):
    column_list = ['Total generation', 
                   'Balearic link',
                    'Peninsular demand', 
                    'Total generation', 
                    'Nuclear', 
                    'Pumped hydro',
                    'Combined Gas Cycle', 
                    'Photovoltaic', 
                    'Solar thermal',
                    'Ocean/Geothermal', 
                    'Fuel', 
                    'Biomass', 
                    'Biogas', 
                    'Hydro', 
                    'Wind',
                    'Natural Gas Cogeneration', 
                    'Waste RSU', 
                    'Coal',
                    'Import France',
                    'Import Portugal', 
                    'Pump Consumption',
                    'Balearic link',
                    'Morocco Exchange',
                    'Portugal Exchange',
                    'France Exchange',
                    'Interconnections exchange']


    tech = []
    for col in column_list:
        if col not in df.columns:
            #tech = col
            #tech_no_available = tech.append()
            print(col)
            df[col] = numpy.nan
            
    df = df.fillna(0)  
    return df  

def renewables(df):
     #map renewables
    renewables = {'Wind', 
                  'Photovoltaic', 
                  'Solar thermal', 
                  'Biomass',
                  'Biogas', 
                  'Waste RSU', 
                  'Hydro', 
                  'Pumped hydro',
                  'Ocean/Geothermal', 
                }
    nonrenewables = {'Nuclear', 
                    'Coal', 
                    'Natural Gas Cogeneration', 
                    'Fuel', 
                    'Combined Gas Cycle'
                    }
    
    #RES-E-RATIO
    df['Renewables'] = ""
    df['NonRenewables'] = ""
    df['Renewables'] = df[(renewables)].sum(axis=1)
    df['NonRenewables'] = df[(nonrenewables)].sum(axis=1)
    df['Total'] = df[['Renewables', 'NonRenewables']].sum(axis=1)
    df['RES-E-RATIO'] = df['Renewables']/df['Total']
    
    return df    
    
def average_carbon_emissions(df):
    EF = {'ef_nuclear': 0.012, 'ef_coal': 1.210, 'ef_combined gas cycle': 0.492, 'ef_cogeneration': 0.380, 'ef_fuel': 0.866,
        'ef_wind': 0.014, 'ef_photovoltaic': 0.071, 'ef_solar thermal': 0.027, 'ef_ocean/geothermal': 0.082, 'ef_biomass':0.054,
        'ef_biogas':0.018,'ef_waste': 0.240, 'ef_hydro':0.024, 'ef_pumped hydro':0.062, 
        'ef_france': 0.068, 'ef_portugal': 0.484, 'ef_morocco': 0.729}

    #final energy values
    mapping_cfs = {'CO2_nuclear': ('Nuclear', 'ef_nuclear'), 
                'CO2_coal': ('Coal', 'ef_coal'), 
                'CO2_combinedgas': ('Combined Gas Cycle', 'ef_combined gas cycle'), 
                'CO2_cogeneration': ('Natural Gas Cogeneration', 'ef_cogeneration'), 
                'CO2_fuel': ('Fuel', 'ef_fuel'), 
                'CO2_wind': ('Wind', 'ef_wind'), 
                'CO2_photovoltaic': ('Photovoltaic', 'ef_photovoltaic'), 
                'CO2_solarthermal': ('Solar thermal', 'ef_solar thermal'),
                'CO2_oceangeothermal': ('Ocean/Geothermal', 'ef_ocean/geothermal'), 
                'CO2_biomass': ('Biomass', 'ef_biomass'),
                'CO2_biogas': ('Biogas', 'ef_biogas'), 
                'CO2_waste': ('Waste RSU', 'ef_waste'), 
                'CO2_hydroUGH': ('Hydro', 'ef_hydro'),
                'CO2_pumpedhydro': ('Pumped hydro', 'ef_pumped hydro'),
                'CO2_france': ('Import France', 'ef_france'), 
                'CO2_portugal': ('Import Portugal', 'ef_portugal')}

    co2_df = df.copy(deep=False)
    for column, data in mapping_cfs.items():
        co2_df[column] = co2_df[data[0]] * EF[data[1]]

    co2_df['Total Emisions'] = co2_df.drop(['Total generation', 
                                            'Balearic link',
                                            'Interconnections exchange',
                                            'Peninsular demand', 
                                            'Total generation', 
                                            'Nuclear', 
                                            'Pumped hydro',
                                            'Combined Gas Cycle', 
                                            'Photovoltaic', 
                                            'Solar thermal',
                                            'Ocean/Geothermal', 
                                            'Fuel', 
                                            'Biomass', 
                                            'Biogas', 
                                            'Hydro', 
                                            'Wind',
                                            'Natural Gas Cogeneration', 
                                            'Waste RSU', 
                                            'Coal',
                                            'Import France',
                                            'Import Portugal', 
                                            'Renewables',
                                            'NonRenewables', 
                                            'Total', 
                                            'RES-E-RATIO',
                                            'Pump Consumption',
                                            'Balearic link',
                                            'Morocco Exchange',
                                            'Portugal Exchange',
                                            'France Exchange',
                                            'Interconnections exchange'], 
                                           axis=1).sum(axis=1)

    co2_df['AEF'] = co2_df['Total Emisions']/co2_df['Peninsular demand']
    df['AEF'] = co2_df['AEF']
    df['Total Emisions'] = co2_df['Total Emisions']
    
    return df 

def average_primary_energy(df):
    PEF = {'pef_nuclear': 3.030, 'pef_coal': 2.790, 'pef_combined gas cycle': 1.970, 'pef_cogeneration': 1.860, 
    'pef_fuel': 2.540, 'pef_wind': 0.030, 'pef_photovoltaic': 0.250, 'pef_solar thermal': 0.030,
    'pef_ocean/geothermal': 0.078, 'pef_biomass': 1.473,'pef_biogas': 2.790,'pef_waste': 1.473, 
    'pef_hydro': 0.100, 'pef_pumped hydro': 1.690, 'pef_france':2.553, 
    'pef_portugal': 1.587, 'pef_morocco': 2.200, 'pef_link': 0.340}

    #primary energy values
    mapping_cfs = {'PE_nuclear': ('Nuclear', 'pef_nuclear'), 
                'PE_coal': ('Coal', 'pef_coal'), 
                'PE_combinedgas': ('Combined Gas Cycle', 'pef_combined gas cycle'), 
                'PE_cogeneration': ('Natural Gas Cogeneration', 'pef_cogeneration'), 
                'PE_fuel': ('Fuel', 'pef_fuel'), 
                'PE_wind': ('Wind', 'pef_wind'), 
                'PE_photovoltaic': ('Photovoltaic', 'pef_photovoltaic'), 
                'PE_solarthermal': ('Solar thermal', 'pef_solar thermal'),
                'PE_oceangeothermal': ('Ocean/Geothermal', 'pef_ocean/geothermal'), 
                'PE_biomass': ('Biomass', 'pef_biomass'),
                'PE_biogas': ('Biogas', 'pef_biogas'), 
                'PE_waste': ('Waste RSU', 'pef_waste'), 
                'PE_hydroUGH': ('Hydro', 'pef_hydro'),
                'PE_pumpedhydro': ('Pumped hydro', 'pef_pumped hydro'),
                'PE_france': ('Import France', 'pef_france')
                }

    PE_df = df.copy(deep=False)
    for column, data in mapping_cfs.items():
        PE_df[column] = PE_df[data[0]] * PEF[data[1]]

    PE_df['Total PE USE'] = PE_df.drop(['Total generation', 
                                            'Balearic link',
                                            'Interconnections exchange',
                                            'Peninsular demand', 
                                            'Total generation', 
                                            'Nuclear', 
                                            'Pumped hydro',
                                            'Combined Gas Cycle', 
                                            'Photovoltaic', 
                                            'Solar thermal',
                                            'Ocean/Geothermal', 
                                            'Fuel', 
                                            'Biomass', 
                                            'Biogas', 
                                            'Hydro', 
                                            'Wind',
                                            'Natural Gas Cogeneration', 
                                            'Waste RSU', 
                                            'Coal',
                                            'Import France',
                                            'Import Portugal', 
                                            'Renewables',
                                            'NonRenewables', 
                                            'Total', 
                                            'RES-E-RATIO',
                                            'Pump Consumption',
                                            'Balearic link',
                                            'Morocco Exchange',
                                            'Portugal Exchange',
                                            'France Exchange',
                                            'Interconnections exchange'], 
                                           axis=1).sum(axis=1)

    PE_df['APEF'] = PE_df['Total PE USE']/PE_df['Peninsular demand']
    
    df['APEF'] = PE_df['APEF']
    df['Total PE USE'] = PE_df['Total PE USE']   
    
    
    return df 

def marginal_signals(df):
    marginal = df[['Total generation', 'Peninsular demand']]
    res = df[['RES-E-RATIO']]
    co2 = df[['Total Emisions','AEF']]
    pe = df[['Total PE USE','APEF']]

    marginal_df = pd.concat([marginal, co2, pe, res], axis=1).fillna(0) 
    marginal_df["DeltaLoad"] = marginal_df["Peninsular demand"].diff()
    marginal_df["DeltaCO2"] = (marginal_df["Peninsular demand"]*marginal_df['AEF']).diff()
    marginal_df["DeltaPE"] = (marginal_df["Peninsular demand"]*marginal_df['APEF']).diff()
    marginal_df = marginal_df.fillna(0)
    #marginal_df.to_csv('marginal_df.csv')
    
    marginal_df['TGC'] = marginal_df['Peninsular demand']/1000
    marginal_df['RES'] = marginal_df['RES-E-RATIO']
    marginal_df['MEFmodel'] = numpy.nan
    marginal_df['MPEFmodel'] = numpy.nan
    # MEF coefficients
    a = [0.5705,-1.2236,0.0054,-0.0335,-0.00036,0.0300]
    for i in range(0,len(df)):
        load=marginal_df.iloc[i].TGC
        res=marginal_df.iloc[i].RES
        MEFmodelled=a[0]+a[1]*res+a[2]*load+a[3]*res*res+a[4]*load*load+a[5]*load*res
        marginal_df.MEFmodel.iloc[i]=MEFmodelled
    
    
    # MPEF coefficients
    b = [-1.6378,-1.1774,0.22068,-1.2180,-0.00384,0.02965]
    for i in range(0,len(df)):
        load=marginal_df.iloc[i].TGC
        res=marginal_df.iloc[i].RES
        MPEFmodelled=b[0]+b[1]*res+b[2]*load+b[3]*res*res+b[4]*load*load+b[5]*load*res
        marginal_df.MPEFmodel.iloc[i]=MPEFmodelled
    
    #df.to_csv('penalty_signals.csv')
    
    return marginal_df 

### ROUTES 
@app.route('/', methods=['POST','GET'])
def home():
    #main
    df = pd.DataFrame(columns=["PBF_shortname","PBF_value", "PBF_datetime"])
    values_df = get_values(df)
    datetime_df = set_datetime_index(values_df)
    sorted_df = pivot_sort_and_fill(datetime_df)
    rename_df = rename_en(sorted_df)
    all_df = build_df(rename_df)
    analysis_df = renewables(all_df)
    co2_df = average_carbon_emissions(analysis_df)
    pe_df = average_primary_energy(co2_df)
    marginal_df = marginal_signals(pe_df)
    #print(pe_df.head())
    #response = analysis_df.to_json(orient ='index')
    #return response

    #result = analysis_df.filter(['Renewables','NonRenewables',
    #                             'Import France','Import Portugal','Import Morocco',
    #                             'Pump Consumption','Balearic Link',
    #                             'Morocco Exchange','Portugal Exchange','France Exchange',
    #                             ])
    
    #result = sorted_df.drop(['Demanda Peninsular','Generación PBF total'], axis=1)
    #.filter(['MEFmodel', 'MPEFmodel'])
    
    #result = marginal_df.filter(['AEF','APEF','MEFmodel','MPEFmodel'])
    #result = analysis_df.filter(['Total','Renewables'])
    
    result = marginal_df
    #result = build_df.filter(['Renewables','NonRenewables'])
     
    #chart_from_python=my_plot_full_bar(result)
    #chart_from_python=my_plot_full(result)
    chart_from_python=dropdown_menu_line(result)
    
    return render_template('newindex.html',chart_for_html=chart_from_python)

@app.route('/dashboard',methods=['POST','GET'])
def dashboard():
    df = pd.DataFrame(columns=["PBF_shortname","PBF_value", "PBF_datetime"])
    values_df = get_values(df)
    datetime_df = set_datetime_index(values_df)
    sorted_df = pivot_sort_and_fill(datetime_df)
    rename_df = rename_en(sorted_df)
    all_df = build_df(rename_df)
    analysis_df = renewables(all_df)
    
    result = analysis_df
    chart_from_python=pie_subplots(result)

    return render_template('newindex.html',chart_for_html=chart_from_python)

app.run()
