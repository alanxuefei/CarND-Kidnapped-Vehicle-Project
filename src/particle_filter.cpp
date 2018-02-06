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

#include "particle_filter.h"

using namespace std;

// declare a random engine to be used across multiple and various method calls
default_random_engine random_engine;

void ParticleFilter::init(double gps_x, double gps_y, double gps_theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 100;

  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  normal_distribution<double> dist_x(gps_x, std_x);
  normal_distribution<double> dist_y(gps_y, std_y);
  normal_distribution<double> dist_theta(gps_theta, std_theta);


  for (int i = 0; i < num_particles; ++i) {
    double sample_x, sample_y, sample_theta;

     sample_x = dist_x(random_engine);
     sample_y = dist_y(random_engine);
     sample_theta = dist_theta(random_engine);

     particles.push_back(Particle{i,sample_x, sample_y, sample_theta, 0, {}, {}, {}});
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // define normal distributions for sensor noise
   normal_distribution<double> noise_x(0, std_pos[0]);
   normal_distribution<double> noise_y(0, std_pos[1]);
   normal_distribution<double> noise_theta(0, std_pos[2]);

   for (int i = 0; i < num_particles; i++) {

     // calculate new state
     if (fabs(yaw_rate) < 0.00001) {
       particles[i].x += velocity * delta_t * cos(particles[i].theta);
       particles[i].y += velocity * delta_t * sin(particles[i].theta);
     }
     else {
       particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
       particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
       particles[i].theta += yaw_rate * delta_t;
     }

     // add noise
     particles[i].x += noise_x(random_engine);
     particles[i].y += noise_y(random_engine);
     particles[i].theta += noise_theta(random_engine);
   }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.

  for (LandmarkObs& observation : observations) {

    double min_dist = numeric_limits<double>::max();
    observation.id = -1;
    
    for (LandmarkObs& predict : predicted) {          
      double cur_dist = dist(observation.x, observation.y, predict.x, predict.y); 
      if (cur_dist < min_dist) {
        min_dist = cur_dist;
        observation.id = predict.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html

  // for each particle...
  for (Particle& particle : particles) {

    double p_x = particle.x;
    double p_y = particle.y;
    double p_theta = particle.theta;

    vector<LandmarkObs> predictions;

    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;
      
      if (dist(lm_x, lm_y, p_x, p_y) <= sensor_range) {

        predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
      }
    }

    vector<LandmarkObs> global_observations;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
      double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      global_observations.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
    }

    dataAssociation(predictions, global_observations);

    particle.weight = 1.0;

    for (unsigned int j = 0; j < global_observations.size(); j++) {
      
      double o_x = global_observations[j].x;
      double o_y = global_observations[j].y;      
      double pr_x, pr_y;

      for (unsigned int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == global_observations[j].id) {
          pr_x = predictions[k].x;
          pr_y = predictions[k].y;
        }
      }

      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(pr_x-o_x,2)/(2*pow(s_x, 2)) + (pow(pr_y-o_y,2)/(2*pow(s_y, 2))) ) );

      particle.weight *= obs_w;
    }
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  vector<Particle> new_particles;
  vector<double> weights;
  double total_weight = 0;
  
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
    total_weight += particles[i].weight;
  }

  uniform_real_distribution<double> unirealdist(0.0, total_weight);

  double beta = 0.0;
  int index = 0;

  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(random_engine);
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
  
