import pandas as pd
import googlemaps
from geopy import distance

print("Geocoding neighborhoods...")

list_of_neighborhoods = pd.read_csv("./data/list_of_neighborhoods.csv",
                                    header=None)
list_of_neighborhoods.columns = ["neighborhood"]

gmaps = googlemaps.Client(key="YOUR API KEY")

# we get the coordinates of the city center
city_center_coordinates = gmaps.geocode("Vitoria, Brazil")
center_lat = city_center_coordinates[0]["geometry"]["location"]["lat"]
center_lon = city_center_coordinates[0]["geometry"]["location"]["lng"]

# then we query every neighborhoods
list_lat = []
list_lon = []
for neighborhood in list_of_neighborhoods.neighborhood:
    # Api documentation: https://developers.google.com/maps/documentation/geocoding/intro#GeocodingResponses
    # some neighborhoods in the dataset do not return their actual location if queried followed by 
    # "Vitoria, Brazil". In those cases, the API returns the coordinates of Vitoria. 
    # A quick hacky workaround to this is check if the lat and lon are exactly matching the ones
    # of Vitoria, to re-query the location without Vitoria in the query text
    try:
        coordinates = gmaps.geocode(neighborhood.lower() + " Vitoria, Brazil") # set the address
        location = coordinates[0]["geometry"]["location"]
        lat, lon = location["lat"],location["lng"]
        assert lat != center_lat and lon != center_lon
    except AssertionError:
        coordinates = gmaps.geocode(neighborhood.lower() + " Espírito Santo, Brazil") # set the address
        location = coordinates[0]["geometry"]["location"]
        lat, lon = location["lat"],location["lng"]
    list_lat.append(lat)
    list_lon.append(lon) 

list_of_neighborhoods["lat"] = list_lat 
list_of_neighborhoods["lon"] = list_lon

# calculate the distances with the distance module of the geopy package
distance_from_center = []
distance_from_center_lat = [] # fixing longitude and looking at east-west distance
distance_from_center_lon = [] # fixing latitude and looking at north-south distance
for row in list_of_neighborhoods.itertuples(index=False):
    distance_from_center.append(
        distance.distance((center_lat,center_lon),(row.lat,row.lon)).km
    )
    distance_from_center_lat.append(
        distance.distance((center_lat,center_lon),(row.lat,center_lon)).km
    )
    distance_from_center_lon.append(
        distance.distance((center_lat,center_lon),(center_lat,row.lon)).km
    )

list_of_neighborhoods["distanceFromCenter"] = distance_from_center
list_of_neighborhoods["distanceFromCenterLat"] = distance_from_center_lat
list_of_neighborhoods["distanceFromCenterLon"] = distance_from_center_lon

# write to disk
list_of_neighborhoods.to_csv("./data/geocoded_neighborhoods.csv",index=False,header=True)