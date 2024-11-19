import React from 'react'

import CategoryWrapper from '../category/CategoryWrapper';
import FeaturedSection from './FeaturedSection';
import LatestRecipe from './LatestRecipe';
import NewsLetter from './NewsLetter';
import Hero from '../../components/Hero';
import AboutSection from './AboutSection';
import CompanyLogo from './CompanyLogo';
import Contact from './Contact';



const Home = () => {
  return (
    <div className="container m-auto">
    <div className="flex flex-col items-center justify-center w-full py-20 ">
      <Hero/>
      <CategoryWrapper/>
    
    </div>

    {/* featured */}
    <FeaturedSection/>
    <LatestRecipe/>
    <NewsLetter/>
    <AboutSection/>
    <CompanyLogo/>
    <Contact/>
   </div>
  )
}

export default Home