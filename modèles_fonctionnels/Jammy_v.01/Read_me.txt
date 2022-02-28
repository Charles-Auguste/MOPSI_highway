____________Jammy modèle v01____________


Description :

	Premier apprentissage après les réglages de bon fonctionnement
	La voiture circule sur une route sans autres véhicules
	Nombre d'itérations effectuées :  500k


Caractéristiques :

	Coefficients d'apprentissage 	-> speed = 5.0
				     		-> action = 1.0
						-> lane_centering = 4.0
						-> Collision_reward = -1.0

	learning rate 			-> 1e-3

	Reward maximum théorique 	-> 8.0
	

Observations :

	Après 500k itérations, le véhicule est capable de suivre le cercle. Cependant,
	on constate de sévères variations de la vitesse sur certains tronçons. De plus
	le véhicule oscille légèrement autour de la route. Vitesse réduite

