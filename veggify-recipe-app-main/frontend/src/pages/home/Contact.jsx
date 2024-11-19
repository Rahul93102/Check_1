import React from 'react'
import { FaCalendar, FaHandRock } from 'react-icons/fa'

const Contact = () => {
  return (
    <div className=" bg-white py-16 rounded-t-md">
      <div className="mx-auto max-w-screen-xl px-6 lg:px-8 mb-20">
      <div className="flex flex-col md:flex-row justify-between items-center gap-20">
            <div className='md:w-1/2'>
            <h2 className="text-3xl font-bold tracking-tight text-secondary sm:text-4xl">Subscribe to our newsletter.</h2>
            <p className="mt-4 text-lg leading-8">
              Nostrud amet eu ullamco nisi aute in ad minim nostrud adipisicing velit quis. Duis tempor incididunt
              dolore.
            </p>
            </div>
            <div className="sm:w-1/2 mt-6 flex flex-col sm:flex-row gap-4">
              <label htmlFor="email-address" className="sr-only">
                Email address
              </label>
              <input
                id="email-address"
                name="email"
                type="email"
                autoComplete="email"
                required
                className="flex-auto rounded-md border-0 bg-primary px-3.5 py-4 text-white shadow-sm  sm:text-sm sm:leading-6 focus:outline-btnColor"
                placeholder="Enter your email"
              />
              <button
                type="submit"
                className="flex-none rounded-md bg-btnColor px-8 py-4 text-sm font-semibold text-white shadow-sm hover:bg-white hover:text-btnColor hover:border-btnColor hover:border"
              >
                Subscribe
              </button>
            </div>
          </div>
      </div>
      <hr />
    </div>
  )
}

export default Contact