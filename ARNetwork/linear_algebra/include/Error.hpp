#pragma once

#include <iostream>

class	Error : public std::exception
{
	private:
		std::string	_message;
	
	public:
		inline		Error(const std::string& message) : _message(message) {}
		inline		~Error(void) throw() {}

		const char	*what(void) const throw() { return _message.c_str(); }
};