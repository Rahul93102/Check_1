import React from 'react'
import { HiOutlineBars3BottomRight } from "react-icons/hi2";
import { RiCloseCircleLine } from "react-icons/ri";


const MobileNav = ({ menuItems, logo, onClose, onOpen, hideLeft }) => {
  return (
    <>
    <div className="h-16 flex justify-between items-center px-6 lg:px-12">
        <a href="/">
          <img src={logo} alt="flex site logo" />
        </a>
        <button className="border border-primary rounded" onClick={onOpen}>
        <HiOutlineBars3BottomRight className='w-7 h-6' />
        </button>
      </div>
      <div
        className={`transition-all w-full h-full fixed bg-primary z-50 top-0 ${hideLeft} flex justify-center items-center`}
      >
        <button className="absolute right-8 top-32" onClick={onClose}>
        <RiCloseCircleLine className='w-7 h-6'/>
        </button>
        <div>
          <ul className="flex flex-col gap-5 ">
            {menuItems?.map((menu, index) => (
              <li key={index}>
                <a
                  href="#!"
                  className="font-medium capitalize text-secondary text-2xl"
                  onClick={onClose}
                >
                  {menu}
                </a>
              </li>
            ))}
          </ul>
          <ul className="flex items-center gap-4 font-medium mt-10">
            <li>
              <button className="text-secondary px-4 py-2 rounded ">Log In</button>
            </li>
            <li>
              <button className="bg-primary text-secondary px-4 py-2 rounded border">
                Sign Up
              </button>
            </li>
          </ul>
        </div>
      </div>
      </>
  )
}

export default MobileNav