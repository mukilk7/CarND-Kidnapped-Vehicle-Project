/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>
#include <cstdio>

#include "particle_filter.h"

using namespace std;

//#define DEBUGGING 1

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	weights.reserve(num_particles);
	particles.reserve(num_particles);

	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		weights[i] = 1.0;
		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle p = particles[i];
		if (fabs(yaw_rate) > 0.0001) {
			p.x += ((velocity / yaw_rate) * (sin(p.theta + delta_t * yaw_rate) - sin(p.theta)));
			p.y += ((velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + delta_t * yaw_rate)));
			p.theta += (yaw_rate * delta_t);
		} else {
			p.x += (velocity * delta_t * cos(p.theta));
			p.y += (velocity * delta_t * sin(p.theta));
		}
		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);
		particles[i] = p;
#ifdef DEBUGGING
		printf("Predicted Particle %d = (%f, %f, %f)\n", i, p.x, p.y, p.theta);
#endif
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {
		LandmarkObs tobs = observations[i];
		double mindist = -1;
		for (int j = 0; j < predicted.size(); j++) {
			LandmarkObs pobs = predicted[j];
			double edist = dist(tobs.x, tobs.y, pobs.x, pobs.y);
			if (mindist < 0 || edist < mindist) {
				tobs.id = pobs.id;
				mindist = edist;
			}
		}
		observations[i] = tobs;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//Store id to landmark for performance
	map<int, Map::single_landmark_s> id2landmark;
	for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
		Map::single_landmark_s l = map_landmarks.landmark_list[j];
		id2landmark[l.id_i] = l;
	}

	double sumwts = 0.0;

	for (int i = 0; i < num_particles; i++) {
		Particle p = particles[i];
#ifdef DEBUGGING
		printf("*************** Particle %d ***********************\n", i);
#endif
		//Compute Observations Translated to Map Coordinates
		vector<LandmarkObs> translated_observations;
		for (int j = 0; j < observations.size(); j++) {
			LandmarkObs obs = observations[j];
			LandmarkObs tobs;
			tobs.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
			tobs.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
			tobs.id = obs.id;
			translated_observations.push_back(tobs);
		}
		//Get the set of landmarks within sensor range
		vector<LandmarkObs> landmarks_in_range;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			Map::single_landmark_s l = map_landmarks.landmark_list[j];
			if (fabs(l.x_f - p.x) <= sensor_range && fabs(l.y_f - p.y) <= sensor_range) {
				LandmarkObs pobs;
				pobs.x = l.x_f; pobs.y = l.y_f; pobs.id = l.id_i;
				landmarks_in_range.push_back(pobs);
			}
		}
		//Do observation association with map landmarks
		dataAssociation(landmarks_in_range, translated_observations);
		//Update particle weight using multivariate gaussian dist
		double twt = 1.0;
		double c = 2 * M_PI * std_landmark[0] * std_landmark[1];
		for (int j = 0; j < translated_observations.size(); j++) {
			LandmarkObs tobs = translated_observations[j];
			Map::single_landmark_s pobs = id2landmark[tobs.id];
			/*
			printf("UW Observation %d -> %d, (%0.2f, %0.2f), (%0.2f, %0.2f)\n", j, tobs.id,
					tobs.x, tobs.y, pobs.x_f, pobs.y_f);
			*/
			double xdiff = (tobs.x - pobs.x_f);
			double ydiff = (tobs.y - pobs.y_f);
			double xfac = (xdiff * xdiff) / (2 * std_landmark[0] * std_landmark[0]);
			double yfac = (ydiff * ydiff) / (2 * std_landmark[1] * std_landmark[1]);
			double wt = exp(-(xfac + yfac));
			wt = wt / c;
			//printf("	xfac = %f, yfac = %f, c = %f, wt = %f, twt = %lf\n", xfac, yfac, c, wt, twt);
			twt *= wt;
		}
		p.weight = twt;
		weights[i] = twt * 100000;
		particles[i] = p;
		sumwts += twt;
#ifdef DEBUGGING
		printf("Total Particle %d Weight = %lf\n", i, twt);
#endif
	}
}

void ParticleFilter::resample() {
#ifdef DEBUGGING
	std::cout << "Resampling particles..." << std::endl;
#endif
	vector<Particle> resampled;
	default_random_engine gen;
	double maxw = *max_element(weights.begin(), weights.end());
	uniform_real_distribution<double> urd(0.0, maxw);
	uniform_int_distribution<int> uid(0, num_particles - 1);
	int index = uid(gen);
	double beta = 0;
	for (int i = 0; i < num_particles; i++) {
		beta += (urd(gen) * 2.0);
		while(weights[index] < beta) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		Particle p = particles[index];
		resampled.push_back(p);
#ifdef DEBUGGING
		printf("	Position %d = Particle %d\n", i, p.id);
#endif
	}
	particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
